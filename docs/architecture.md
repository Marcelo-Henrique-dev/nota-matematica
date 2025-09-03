# Arquitetura do Projeto

```mermaid
flowchart TD
    A[Arquivo CSV: students.csv] --> B[Ingestão de Dados (pandas)]
    B --> C[Processamento e Feature Engineering]
    C --> D[Criação de Database MySQL]
    D --> E[Execução dos scripts SQL (SOR/SOT/SPEC)]
    E --> F[Inserção dos dados nas tabelas]
    F --> G[Treinamento do Modelo (LinearRegression)]
    G --> H[Persistência do Modelo (.pickle na pasta model)]
    H --> I[Drop do Database após tratamento]
```

## Descrição das Etapas

- **A:** Fonte dos dados brutos (SOR).
- **B:** Leitura dos dados via pandas.
- **C:** Seleção de colunas e transformação (one-hot encoding).
- **D:** Criação do banco de dados MySQL.
- **E:** Execução dos scripts SQL para criar tabelas SOR, SOT e SPEC.
- **F:** Inserção dos dados do CSV nas tabelas do banco.
- **G:** Treinamento do modelo de regressão linear.
- **H:** Salvamento do modelo treinado como arquivo pickle na pasta `model`.
- **I:** Remoção do banco de dados após o processamento.

---

> O fluxo garante rastreabilidade dos dados, automação do pipeline e descarte seguro do ambiente de dados