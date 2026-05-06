import chainlit as cl

from mvc.config import OPENROUTER_API_KEY, MODEL_NAME, CHROMA_DIR, EMBEDDING_MODEL
from mvc.model import Workflow
from mvc.services.llm import new_llm
from mvc.services.db import VectorDatabase


@cl.on_chat_start
async def on_chat_start():
    llm = new_llm(
        model=MODEL_NAME,
        openrouter_api_key=OPENROUTER_API_KEY,
        streaming=True,
    )

    db = VectorDatabase(
        chroma_dir=CHROMA_DIR,
        embedding_model=EMBEDDING_MODEL,
    )

    workflow = Workflow(llm=llm, db=db)
    cl.user_session.set("workflow", workflow)


@cl.on_message
async def on_message(message: cl.Message):
    workflow: Workflow = cl.user_session.get("workflow")

    msg = cl.Message(content="")

    async for chunk in workflow.astream(message.content):
        await msg.stream_token(chunk)

    await msg.send()