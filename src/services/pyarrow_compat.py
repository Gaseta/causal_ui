"""
Inicialização eager e thread-safe do PyArrow.

Este módulo deve ser importado o **mais cedo possível** no processo Python,
antes de qualquer outro módulo que possa puxar PyArrow indiretamente
(Streamlit, pandas, dowhy, econml). Importe-o no topo do ``app.py``:

    # Primeira linha do app.py, antes de qualquer outro import.
    from src.services import pyarrow_compat  # noqa: F401

## Por que isto existe

PyArrow em ARM64 tem dois bugs nativos conhecidos que se manifestam como
SIGSEGV em workloads que misturam threads + chamadas concorrentes a
``pyarrow.compute``:

1. **Race no function registry** ([ARROW-10784]). A primeira chamada a
   ``pyarrow.compute`` inicializa um hashtable global. Se duas threads
   tentarem inicializar simultaneamente, uma pode ler entries inválidas
   durante o rehash, causando SIGSEGV em
   ``_Hashtable::_M_find_before_node``.

2. **Allocator interno (jemalloc) com background thread** que corrompe heap
   em padrões específicos de aarch64 + cgroup ([arrow#44342]).

## O que este módulo faz

1. Importa ``pyarrow.compute`` **single-threaded** (na thread principal,
   ANTES do Streamlit subir suas threads de IO). Isso garante que o
   function registry é populado uma vez só, sem race.

2. Verifica se as variáveis de ambiente que limitam threads do Arrow estão
   setadas. Se não estiverem, define defaults seguros via runtime API.

## Quando não usar

Se você está em x86_64 com hardware moderno e não vê crashes, este módulo
é desnecessário (e adiciona ~200ms ao startup). Mas é seguro deixar
importado — não causa side effects negativos.
"""
from __future__ import annotations

import os
import sys


def _set_env_if_unset(key: str, value: str) -> None:
    """Define ``os.environ[key] = value`` apenas se ainda não estiver setado."""
    if key not in os.environ:
        os.environ[key] = value


# ----------------------------------------------------------------------
# 1. Variáveis de ambiente que limitam threads em libs nativas.
# ----------------------------------------------------------------------
# Estes precisam estar setados ANTES do import de pyarrow. Em produção é
# preferível setá-los no Dockerfile/docker-compose; aqui é fallback para
# desenvolvimento local.
_set_env_if_unset("OMP_NUM_THREADS", "1")
_set_env_if_unset("OPENBLAS_NUM_THREADS", "1")
_set_env_if_unset("MKL_NUM_THREADS", "1")
_set_env_if_unset("ARROW_NUM_THREADS", "1")

# Desabilita o background thread do jemalloc no PyArrow (causa de
# corrupção de heap em alguns ambientes ARM64).
_set_env_if_unset("MALLOC_ARENA_MAX", "1")


# ----------------------------------------------------------------------
# 2. Importação eager de pyarrow + pyarrow.compute (thread principal,
#    sem concorrência), para popular o function registry uma vez só.
# ----------------------------------------------------------------------
def _eager_init_pyarrow() -> None:
    """Importa pyarrow.compute na thread atual, populando o registry global.

    Se algum import falhar (PyArrow não instalado), apenas avisa e segue —
    o app pode rodar sem pyarrow desde que não chame ``st.dataframe``.
    """
    try:
        import pyarrow  # noqa: F401
        import pyarrow.compute as pc  # noqa: F401

        # Força a inicialização do registry global de funções. Sem esta linha,
        # o registry é populado lazy na primeira chamada de qualquer função.
        # Com esta linha, populamos AGORA na thread principal, antes que o
        # Streamlit suba threads de IO.
        registry = pc.function_registry()
        _ = registry.list_functions()

        # Aplica limite de threads via API runtime, caso a ENV não tenha
        # pego (a ENV só funciona se setada antes do load da libarrow).
        try:
            pyarrow.set_cpu_count(1)
            pyarrow.set_io_thread_count(1)
        except Exception:
            pass

    except ImportError:
        print(
            "[pyarrow_compat] PyArrow não está instalado; pulando inicialização. "
            "Tabelas do Streamlit podem não funcionar.",
            file=sys.stderr,
            flush=True,
        )
    except Exception as exc:
        # Não queremos derrubar o app se a inicialização eager falhar — o
        # importante é tentar. Se der erro, registramos mas continuamos.
        print(
            f"[pyarrow_compat] Aviso: falha ao inicializar PyArrow: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )


_eager_init_pyarrow()