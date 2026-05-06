from telegram.ext import Application, MessageHandler, filters

from mvc.frontends.telegram.bot import Bot
from mvc.config import MODEL_NAME, OPENROUTER_API_KEY, CHROMA_DIR, EMBEDDING_MODEL
from mvc.services.llm import new_llm
from mvc.services.db import VectorDatabase


def main():
    llm = new_llm(
        model=MODEL_NAME,
        openrouter_api_key=OPENROUTER_API_KEY,
        streaming=True,
    )

    db = VectorDatabase(
        chroma_dir=CHROMA_DIR,
        embedding_model=EMBEDDING_MODEL,
    )

    bot = Bot(llm=llm, db=db)

    application = Application.builder().token(bot.get_bot_token()).build()

    application.add_handler(MessageHandler(filters.ALL, bot.on_update_received))

    application.run_polling()


if __name__ == "__main__":
    main()