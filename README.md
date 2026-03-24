# 🧠 Causal Oncology Analysis App

Aplicação em **Python + Streamlit** para construção de pipelines completos de análise causal em dados genômicos:

- 📊 Pré-processamento de dados (MAF + clínico)
- 🔗 Descoberta causal (PC, GES, py-tetrad)
- 📈 Inferência causal (DoWhy + refutações)

---

## 🚀 Funcionalidades

### 📊 1. Pré-processamento
- Upload de arquivos:
  - MAF
  - Clinical Sample
  - Clinical Patient
- Seleção dinâmica de colunas
- Integração com Reactome (pathways + subpathways)
- Geração de:
  - datasets por pathway
  - `CombinedDF.tsv`
  - `group_resume.tsv`

---

### 🔗 2. Descoberta Causal
- Algoritmos disponíveis:
  - PC
  - GES
  - (extensível via py-tetrad)
- Parâmetros configuráveis via UI
- Outputs:
  - grafo causal (DOT)
  - imagem (PNG)
  - edges (TSV)

---

### 📈 3. Inferência Causal
- Integração com **DoWhy**
- Seleção de:
  - treatment
  - outcome
- Execução de:
  - estimadores causais
  - refutadores
- Exportação por combinação

---

## ⚙️ Requisitos

### 🔹 Local
- Python **3.11 (recomendado)**
- Java **17+** (necessário para py-tetrad)

### 🔹 Docker
- Docker Desktop (ou Colima)
- Docker Compose

---

## 🧪 Executando localmente

```bash
# 1. criar ambiente virtual
python3.11 -m venv .venv
source .venv/bin/activate

# 2. atualizar pip
pip install --upgrade pip

# 3. instalar dependências
pip install -r requirements.txt

# 4. rodar aplicação
streamlit run app.py

# 5. acesse:
http://localhost:8501
```

## 🐳 Executando com Docker

### 🔹 1. Build da imagem

```bash
docker compose build
```

### 🔹 2. Rodar aplicação

```bash
docker compose up
```

### 🔹 3. Acessar aplicação

```bash
http://localhost:8501
```

### 🔹 4. Parar aplicação

```bash
docker compose down
```

---

## 🧠 Arquitetura

O projeto segue uma arquitetura modular baseada em separação de responsabilidades:

### 🔹 `domain/`
- modelos de dados  
- configurações  

### 🔹 `services/`
- lógica de negócio  
- pré-processamento  
- descoberta causal  
- inferência causal  

### 🔹 `ui/`
- interface Streamlit  
- controle dos fluxos  

Essa organização permite:
- desacoplamento entre UI e lógica
- maior facilidade de manutenção
- extensão simples de novos algoritmos

---

## ⚠️ Observações importantes

- O suporte ao **py-tetrad** depende de Java + JAR  
- Existe fallback automático para `causal-learn` quando necessário  
- O processamento pode ser intensivo dependendo do número de pathways  
- Recomenda-se iniciar com datasets menores para validação do pipeline  
