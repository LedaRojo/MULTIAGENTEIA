#RAG para ciberseguridad
#se ejecuta en huggingface el sistema pero el despliegue en github


import os
import gradio as gr

print("GRADIO VERSION:", gr.__version__)

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
import tiktoken

# Configuración
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY no encontrada en variables de entorno")

vectorstore = None
retriever = None
chain = None


def initialize_rag_system(files):
    global vectorstore, retriever, chain

    if files is None or len(files) == 0:
        return "⚠️ No se detectaron archivos. Subí al menos un documento antes de inicializar."

    try:
        docs_list = []
        for file_info in files:
            file_path = file_info if isinstance(file_info, str) else getattr(file_info, "name", None)
            if not file_path:
                continue

            loader = UnstructuredFileLoader(file_path)
            documents = loader.load()
            docs_list.extend(documents)

        if not docs_list:
            return "⚠️ Se subieron archivos, pero no pude extraer texto. Probá con PDF no escaneado o .txt."

        _ = tiktoken.encoding_for_model("gpt-3.5-turbo")
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200
        )
        doc_splits = text_splitter.split_documents(docs_list)

        embedding_model = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=OPENAI_API_KEY
        )

        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=embedding_model
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        template = """Eres un asistente útil que responde preguntas basándose ÚNICAMENTE en el contexto proporcionado.

Contexto:
{context}

Pregunta: {question}

Instrucciones:
- Responde de manera clara y concisa
- Si la información no está en el contexto, di "No encontré información sobre esto en los documentos"
- Usa el mismo lenguaje que la pregunta
- Sé específico y evita información irrelevante

Respuesta:"""

        prompt = ChatPromptTemplate.from_template(template)

        model = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            temperature=0,
            model="gpt-4o-mini"
        )

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        return f"✅ Sistema RAG inicializado exitosamente con {len(docs_list)} documentos y {len(doc_splits)} chunks."

    except Exception as e:
        return f"❌ Error inicializando el sistema: {str(e)}"


def ask_question(question, history):
    """History en formato messages: [{'role':..., 'content':...}, ...]"""
    history = history or []

    if chain is None:
        history.append({"role": "assistant", "content": "⚠️ Primero debes subir archivos e inicializar el sistema."})
        return "", history

    if not question or not question.strip():
        history.append({"role": "assistant", "content": "❌ Por favor ingresa una pregunta."})
        return "", history

    try:
        history.append({"role": "user", "content": question})
        respuesta = chain.invoke(question)
        history.append({"role": "assistant", "content": respuesta})
        return "", history

    except Exception as e:
        history.append({"role": "assistant", "content": f"❌ Error procesando la pregunta: {str(e)}"})
        return "", history


def clear_chat():
    return []


with gr.Blocks(title="Sistema RAG - Asistente de Documentos") as demo:
    gr.Markdown(
        """
        # 🤖 Sistema RAG - Asistente Inteligente de Información Empresarial
        **Carga tus documentos y haz preguntas sobre su contenido**
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Configuración")
            file_input = gr.File(
                label="Sube tus documentos",
                file_types=[".txt", ".pdf", ".docx", ".doc"],
                file_count="multiple",
                type="filepath"
            )
            init_button = gr.Button("🚀 Inicializar Sistema RAG", variant="primary")
            init_status = gr.Textbox(label="Estado del Sistema", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Consulta la información que necesitas")
            chatbot = gr.Chatbot(label="Conversación", height=500)
            question_input = gr.Textbox(
                label="Escribe tu pregunta",
                placeholder="¿Qué información buscas en los documentos?",
                lines=2
            )

            with gr.Row():
                submit_btn = gr.Button("📤 Enviar Pregunta", variant="primary")
                clear_btn = gr.Button("🗑️ Limpiar Chat", variant="secondary")

    init_button.click(
        fn=initialize_rag_system,
        inputs=[file_input],
        outputs=[init_status]
    )

    submit_btn.click(
        fn=ask_question,
        inputs=[question_input, chatbot],
        outputs=[question_input, chatbot]
    )

    question_input.submit(
        fn=ask_question,
        inputs=[question_input, chatbot],
        outputs=[question_input, chatbot]
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot]
    )

    gr.Markdown(
        """
        ---
        ### 📝 Formatos soportados:
        - 📄 PDF
        - 📝 Word (.docx, .doc)
        - 📋 Texto (.txt)
        """
    )


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
