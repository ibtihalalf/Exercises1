import io

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from mvc.services.db import VectorDatabase


system_prompt = """
You are a helpful RAG assistant.

Answer the user's question using the provided context.
If the answer is not in the context, say you don't know based on the uploaded documents.
Do not invent information.

Context:
{context}
"""


class Workflow:
    def __init__(self, llm: ChatOpenAI, db: VectorDatabase):
        self.llm = llm
        self.db = db
        self.parser = StrOutputParser()

        self.messages: list[BaseMessage] = [
            SystemMessage("You are a helpful assistant that answers using retrieved documents.")
        ]

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    async def astream(self, message: str):
        self.messages.append(HumanMessage(message))

        docs = self.db.retrieve(message, k=3)
        context = self.format_docs(docs)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{question}"),
            ]
        )

        chain = prompt | self.llm | self.parser

        chunks = []

        async for chunk in chain.astream(
            {
                "question": message,
                "context": context,
            }
        ):
            chunks.append(chunk)
            yield chunk

        self.messages.append(AIMessage("".join(chunks)))

    def messages_to_markdown(self) -> str:
        text = io.StringIO()

        for msg in self.messages:
            if isinstance(msg, HumanMessage):
                who = "**User**"
            elif isinstance(msg, AIMessage):
                who = "**Bot**"
            elif isinstance(msg, SystemMessage):
                who = "**System**"
            else:
                who = f"**{msg.type}**"

            text.write(f"{who}: {msg.content}\n\n")

        return text.getvalue()