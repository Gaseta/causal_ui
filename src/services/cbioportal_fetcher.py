from __future__ import annotations

import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional


CBIOPORTAL_BLOB_BASE = "https://github.com/cBioPortal/datahub/blob/master/public"

# Arquivos que o pipeline de pré-processamento consome.
REQUIRED_FILES = (
    "data_clinical_patient.txt",
    "data_clinical_sample.txt",
    "data_mutations.txt",
)

# User-Agent: alguns CDNs do GitHub rejeitam o default do urllib (Python-urllib/X.Y).
_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; cBioPortalFetcher/1.0; "
        "+https://github.com/cBioPortal/datahub)"
    )
}


ProgressCallback = Optional[Callable[[str, Optional[float]], None]]


@dataclass(slots=True)
class FetchResult:
    """Resultado do fetch de um dataset."""

    dataset: str
    output_dir: Path
    mutations_file: Path
    clinical_sample_file: Path
    clinical_patient_file: Path
    files_already_present: List[str]
    files_downloaded: List[str]


def _normalize_dataset_name(name: str) -> str:
    """Normaliza o nome do dataset para uso na URL.

    cBioPortal Datahub usa nomes em snake_case minúsculo (ex.: brca_metabric,
    luad_tcga, acc_2019). Aceitamos espaços, hífens e variações de caixa;
    convertemos para o formato esperado.
    """
    name = str(name).strip().lower()
    # Espaços e hífens viram underscore.
    name = re.sub(r"[\s\-]+", "_", name)
    # Remove caracteres inválidos para path de URL.
    name = re.sub(r"[^a-z0-9_]", "", name)
    # Colapsa múltiplos underscores.
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def _build_download_url(dataset: str, filename: str) -> str:
    """Constrói URL de download que resolve Git LFS automaticamente.

    Usamos ``https://github.com/<org>/<repo>/blob/<branch>/<path>?raw=true``
    porque o cBioPortal Datahub armazena ``data_mutations.txt`` (e outros
    arquivos grandes) via Git LFS. A URL ``raw.githubusercontent.com``
    retornaria apenas o pointer LFS (algumas centenas de bytes) para esses
    arquivos. A URL com ``?raw=true`` faz o GitHub resolver o pointer e
    servir o conteúdo real do LFS.
    """
    return f"{CBIOPORTAL_BLOB_BASE}/{dataset}/{filename}?raw=true"


def _download_one(
    url: str,
    target_path: Path,
    progress_callback: ProgressCallback = None,
    chunk_size: int = 64 * 1024,
) -> None:
    """Baixa um único arquivo em streaming, com tratamento de erros claros."""
    request = urllib.request.Request(url, headers=_HTTP_HEADERS)

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total = response.headers.get("Content-Length")
            total = int(total) if total and total.isdigit() else None

            tmp_path = target_path.with_suffix(target_path.suffix + ".part")
            tmp_path.parent.mkdir(parents=True, exist_ok=True)

            downloaded = 0
            last_report = 0.0

            with open(tmp_path, "wb") as fh:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)

                    # Reporta progresso no máximo a cada 0.5s para não saturar a UI.
                    now = time.monotonic()
                    if progress_callback is not None and (now - last_report) > 0.5:
                        last_report = now
                        if total:
                            mb = downloaded / (1024 * 1024)
                            total_mb = total / (1024 * 1024)
                            progress_callback(
                                f"Baixando {target_path.name}: "
                                f"{mb:.1f} / {total_mb:.1f} MB",
                                None,
                            )
                        else:
                            mb = downloaded / (1024 * 1024)
                            progress_callback(
                                f"Baixando {target_path.name}: {mb:.1f} MB",
                                None,
                            )

            # Atômico: só move o .part para o nome final quando o download completa.
            tmp_path.replace(target_path)

            # Detecção defensiva de pointer LFS: se a URL não resolveu o
            # conteúdo real do LFS (por algum motivo), o arquivo baixado vai
            # ser muito pequeno (~130 bytes) e começar com "version https://..."
            # Verificamos isso e damos erro claro em vez de deixar o pipeline
            # quebrar com um TSV inválido lá na frente.
            if target_path.stat().st_size < 1024:  # arquivo suspeito de ser pointer
                try:
                    with open(target_path, "rb") as fh:
                        head = fh.read(64)
                    if head.startswith(b"version https://git-lfs"):
                        target_path.unlink(missing_ok=True)
                        raise RuntimeError(
                            f"O download retornou um pointer Git LFS em vez do "
                            f"conteúdo real. URL: {url}\n"
                            f"Isso pode acontecer se o GitHub falhar em resolver "
                            f"o arquivo do LFS. Tente novamente em alguns minutos."
                        )
                except (OSError, IOError):
                    pass

    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise FileNotFoundError(
                f"Arquivo não encontrado no datahub do cBioPortal: {url}\n"
                f"Verifique se o nome do dataset está correto e se ele contém o arquivo "
                f"'{target_path.name}'."
            ) from exc
        raise RuntimeError(
            f"Erro HTTP {exc.code} ao baixar {url}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Erro de rede ao baixar {url}: {exc.reason}.\n"
            f"Verifique sua conexão de internet."
        ) from exc


def fetch_dataset(
    dataset_name: str,
    target_root: Path,
    progress_callback: ProgressCallback = None,
    force_redownload: bool = False,
) -> FetchResult:
    """Baixa os 3 arquivos clínicos/mutacionais de um dataset do cBioPortal.

    Parâmetros
    ----------
    dataset_name : str
        Nome do dataset no datahub (ex.: ``brca_metabric``, ``luad_tcga``,
        ``acc_2019``). Aceita variações de caixa e separadores; será
        normalizado internamente.

    target_root : Path
        Diretório onde os arquivos serão salvos. Os arquivos ficam em
        ``target_root / <dataset_normalizado> / <filename>``.

    progress_callback : callable, opcional
        Função ``(mensagem, progresso)`` que recebe atualizações de progresso.
        ``progresso`` é None ou um float em [0, 1]. Use para integrar com
        Streamlit ou similar.

    force_redownload : bool
        Se True, baixa novamente mesmo que o arquivo já exista. Default False
        (cache local — útil em desenvolvimento iterativo).

    Retorna
    -------
    FetchResult
        Caminhos absolutos dos 3 arquivos baixados, prontos para serem
        passados ao PreprocessingConfig.

    Erros
    -----
    FileNotFoundError
        Se algum dos arquivos não existir no datahub (HTTP 404).

    RuntimeError
        Para erros de rede ou HTTP diferentes de 404.

    ValueError
        Se o nome do dataset for vazio após normalização.
    """
    normalized = _normalize_dataset_name(dataset_name)
    if not normalized:
        raise ValueError(
            f"Nome de dataset inválido: '{dataset_name}'. "
            "Use o identificador do datahub (ex.: brca_metabric, luad_tcga)."
        )

    output_dir = target_root / normalized
    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback is not None:
        progress_callback(
            f"Iniciando fetch do dataset '{normalized}' do cBioPortal Datahub.",
            None,
        )

    already_present: List[str] = []
    downloaded: List[str] = []

    file_paths: Dict[str, Path] = {}

    for filename in REQUIRED_FILES:
        target = output_dir / filename
        file_paths[filename] = target

        if target.exists() and not force_redownload:
            already_present.append(filename)
            if progress_callback is not None:
                progress_callback(
                    f"{filename} já está em cache local — pulando download.",
                    None,
                )
            continue

        url = _build_download_url(normalized, filename)
        if progress_callback is not None:
            progress_callback(f"Baixando {filename} de {url}.", None)

        _download_one(url, target, progress_callback=progress_callback)
        downloaded.append(filename)

    if progress_callback is not None:
        progress_callback(
            f"Fetch concluído: {len(downloaded)} baixado(s), "
            f"{len(already_present)} já em cache. "
            f"Arquivos disponíveis em {output_dir}.",
            None,
        )

    return FetchResult(
        dataset=normalized,
        output_dir=output_dir,
        mutations_file=file_paths["data_mutations.txt"],
        clinical_sample_file=file_paths["data_clinical_sample.txt"],
        clinical_patient_file=file_paths["data_clinical_patient.txt"],
        files_already_present=already_present,
        files_downloaded=downloaded,
    )