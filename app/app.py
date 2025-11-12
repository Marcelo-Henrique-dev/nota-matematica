import sys, os, pickle
import streamlit as st
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.data.readcsv import read_csv
from core.data.database import (
    create_database_and_tables,
    insert_csv_to_sor,
    run_etl_sor_to_sot,
    run_etl_sot_to_spec_train,
    run_etl_for_test_data,
    load_data,
    drop_database
)
from core.features.preprocess import make_preprocess_pipeline
from core.models.train import train_regressor, train_classifier
from core.models.predict import evaluate_classifier, evaluate_regressor
from core.explain.coefficients import extract_logit_importances, extract_linear_importances
from core.chatbot.rules import answer_from_metrics

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

st.set_page_config(page_title="📚 Students Performance - (CHATBOT + IA)", layout="wide")
st.title("📚 Students Performance - CHATBOT")

# -----------------------------
# Inicialização do estado
# -----------------------------
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "predictions_made" not in st.session_state:
    st.session_state.predictions_made = False
if "prediction_df" not in st.session_state:
    st.session_state.prediction_df = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Oi, eu sou o bot do OpenFoodFacts. Envie seus dados para começarmos!"}
    ]
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "importances" not in st.session_state:
    st.session_state.importances = None
if "uploaded_dfs" not in st.session_state:
    st.session_state.uploaded_dfs = {}

MODEL_DIR = "model"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
MODEL_PATH = os.path.join(MODEL_DIR, "off_model.pickle")

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

def get_openai_client():
    key = "KEY AQUI"
    if not key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None

def build_rag_context(df: pd.DataFrame, max_chars: int = 4000) -> str:
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include=['object','category','bool']).columns
    parts = [f"Shape: {df.shape[0]} x {df.shape[1]}"]
    if len(num_cols) > 0:
        desc = df[num_cols].describe().T
        desc['median'] = df[num_cols].median()
        cols = ['count','mean','median','std','min','max']
        parts.append("[Resumo Numérico]\n" + desc[cols].to_string())
    if len(cat_cols) > 0:
        cat_summary = []
        for c in cat_cols:
            vc = df[c].value_counts(dropna=False).head(5)
            cat_summary.append(f"{c}:\n{vc.to_string()}")
        parts.append("[Resumo Categórico]\n" + "\n".join(cat_summary))
    ctx = "\n\n".join(parts)
    return ctx[:max_chars] + ("\n... (contexto truncado)" if len(ctx) > max_chars else "")


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("1. Upload da Base de Dados")
    uploaded_files = st.file_uploader(
        "Envie arquivo csv",
        type=["csv", "tsv"],
        accept_multiple_files=True
    )

    # Lê e guarda DataFrames no session_state
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_dfs:
                df = read_csv(file)
                st.session_state.uploaded_dfs[file.name] = df

    st.header("2. Ações")

    st.subheader("Treinar Novo Modelo")
    task = st.selectbox("Escolha a tarefa", ["Classificação", "Regressão"])
    test_size = st.slider("Tamanho do conjunto de teste", 0.1, 0.4, 0.2, 0.05)

    if st.button("Treinar Modelo"):
        df_train = None
        for name, df in st.session_state.uploaded_dfs.items():
            if name.endswith((".csv", ".tsv")) and "test" not in name.lower():
                df_train = df

        if df_train is not None:
            with st.spinner("Treinando o modelo..."):
                create_database_and_tables()
                insert_csv_to_sor(df_train)
                run_etl_sor_to_sot()
                run_etl_sot_to_spec_train()

                df_spec_train = load_data("spec_math")
                target = "math_score"
                if target not in df_spec_train.columns:
                    st.error(f"A coluna alvo '{target}' não foi encontrada.")
                    st.stop()

                if task == "Classificação":
                    y = (df_spec_train[target] > 10).astype(int)
                    X = df_spec_train.drop(columns=[target])
                    pre = make_preprocess_pipeline(X)
                    model, X_test, y_test = train_classifier(X, y, pre, test_size=test_size)
                    st.session_state.metrics, cm = evaluate_classifier(model, X_test, y_test)
                    st.session_state.importances = extract_logit_importances(model, X)
                else:
                    y = df_spec_train[target]
                    X = df_spec_train.drop(columns=[target])
                    pre = make_preprocess_pipeline(X)
                    model, X_test, y_test = train_regressor(X, y, pre, test_size=test_size)
                    st.session_state.metrics = evaluate_regressor(model, X_test, y_test)
                    st.session_state.importances = extract_linear_importances(model, X.columns, pre)

                with open(MODEL_PATH, "wb") as file:
                    pickle.dump(model, file)

                st.session_state.model_trained = True
                st.session_state.predictions_made = False
            st.success("✅ Modelo treinado e salvo com sucesso!")
        else:
            st.warning("Arquivo csv não encontrado.")

    st.subheader("Usar Modelo Existente")
    if st.button("Carregar Modelo e Fazer Previsões"):
        if not os.path.exists(MODEL_PATH):
            st.error("Nenhum modelo salvo encontrado. Treine um modelo primeiro.")
        else:
            df_test = None
            for name, df in st.session_state.uploaded_dfs.items():
                if name.endswith((".csv", ".tsv")) and "test" not in name.lower():
                    df_test = df

            if df_test is not None:
                with st.spinner("Carregando modelo e fazendo previsões..."):
                    run_etl_for_test_data(df_test)
                    df_spec_predict = load_data("spec_math")

                    with open(MODEL_PATH, "rb") as file:
                        model = pickle.load(file)

                    ids = df_spec_predict[["code"]] if "code" in df_spec_predict.columns else None
                    X_predict = df_spec_predict.drop(columns=["code"], errors="ignore")

                    predictions = model.predict(X_predict)

                    result_df = pd.DataFrame({"prediction": predictions})
                    if ids is not None:
                        result_df.insert(0, "code", ids)

                    st.session_state.prediction_df = result_df
                    st.session_state.predictions_made = True
                st.success("Previsões geradas com sucesso!")
            else:
                st.warning("Arquivo csv não encontrado.")

    st.header("3. Excluir Dados")
    if st.button("Excluir Banco e Modelo e Resetar Sessão"):
        drop_database()
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        st.session_state.clear()
        st.info("Banco, modelo e sessão resetados.")
        st.rerun()

# -----------------------------
# Abas principais
# -----------------------------
tab_train, tab_predict, tab_chat, tab_rag = st.tabs(["📊 Resultados do Treino", "🚀 Previsões", "💬 Chat", "💬 Chat Ia"])

with tab_train:
    st.header("Resultados do Modelo")
    if not st.session_state.model_trained or st.session_state.metrics is None:
        st.info("Treine um modelo para ver resultados.")
    else:
        st.subheader("📈 Métricas")
        st.json(st.session_state.metrics)
        st.subheader("🔎 Importâncias")
        st.dataframe(st.session_state.importances.head(20), use_container_width=True)

with tab_predict:
    st.header("Previsões em Dados de Teste")
    if not st.session_state.predictions_made:
        st.info("Carregue um modelo e faça previsões na barra lateral.")
    else:
        st.dataframe(st.session_state.prediction_df)
        csv_data = convert_df_to_csv(st.session_state.prediction_df)
        st.download_button(
            label="⬇️ Baixar Previsões em CSV",
            data=csv_data,
            file_name="predictions.csv",
            mime="text/csv",
        )

with tab_chat:
    st.header("Converse com o Modelo")
    if not st.session_state.model_trained:
        st.info("Treine um modelo primeiro para poder conversar.")
    else:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if prompt := st.chat_input("Pergunte sobre métricas ou importâncias..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            response = answer_from_metrics(
                question=prompt,
                task=task,
                metrics_df_or_dict=st.session_state.metrics,
                importances_df=st.session_state.importances
            )
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()

with tab_rag:
    st.header("🤖 Chat RAG — Perguntas com Contexto da Base")

    # Carrega o dataset base
    df_rag = None
    for name, df in st.session_state.uploaded_dfs.items():
        if name.endswith((".csv", ".tsv")) and "test" not in name.lower():
            df_rag = df

    if df_rag is None:
        st.info("Faça upload de Train.csv ou StudentsPerformance.csv para usar o chat RAG.")
    else:
        max_ctx_chars = st.slider("Limite do contexto (caracteres)", 500, 12000, 4000, step=500)
        show_rag_ctx = st.checkbox("Mostrar contexto RAG", value=False)

        rag_context = build_rag_context(df_rag, max_chars=max_ctx_chars)
        if show_rag_ctx:
            with st.expander("📊 Ver contexto RAG (resumo da base)"):
                st.text(rag_context)

        if "rag_messages" not in st.session_state:
            st.session_state.rag_messages = [
                {"role": "assistant", "content": "Olá! Eu posso responder perguntas sobre a base de empréstimos com base no contexto gerado."}
            ]

        for msg in st.session_state.rag_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        rag_prompt = st.chat_input("Pergunte algo sobre o conjunto de dados ou as variáveis...")

        if rag_prompt:
            st.session_state.rag_messages.append({"role": "user", "content": rag_prompt})
            messages = [
                {"role": "system", "content": "Você é um professor de matemática. Use o contexto fornecido para responder perguntas sobre os dados."},
                {"role": "user", "content": f"Contexto (resumo da base):\n{rag_context}"}
            ]
            for m in st.session_state.rag_messages:
                if m["role"] in ("user", "assistant"):
                    messages.append(m)

            client = get_openai_client()
            if client:
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        temperature=0.3
                    )
                    reply = response.choices[0].message.content
                except Exception as e:
                    reply = f"⚠️ Erro ao acessar a OpenAI: {e}"
            else:
                reply = f"(Sem OpenAI configurado)\nPergunta: {rag_prompt}\n\nContexto:\n{rag_context[:1000]}..."

            st.session_state.rag_messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.rerun()
