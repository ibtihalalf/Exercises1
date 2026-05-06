from mvc.model import Workflow
from mvc.frontends.cli.view import SupportBotApp
from mvc.services.llm import new_llm
from mvc.services.db import VectorDatabase
from mvc.config import MODEL_NAME, OPENROUTER_API_KEY, CHROMA_DIR, EMBEDDING_MODEL


if __name__ == "__main__":
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

    app = SupportBotApp(workflow=workflow)
    app.run()