import os
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


class RAGSearch:
    """RAG search and summarization using Groq LLM"""

    def __init__(self, vector_store=None, embedding_manager=None):
        """
        Initialize RAG search

        Args:
            vector_store: Vector store instance (optional)
            embedding_manager: Embedding manager instance (optional)
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

        # Initialize Groq LLM
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1024,
        )

        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following context to answer the question concisely and accurately.

Context:
{context}

Question: {question}

Provide a clear and focused answer based only on the context provided. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question based on the provided context."

Answer:""",
        )

    def search_and_summarize(
        self, query: str, top_k: int = 3, vector_store=None
    ) -> str:
        """
        Search for relevant documents and generate summary

        Args:
            query: Search query
            top_k: Number of top documents to retrieve
            vector_store: Override vector store if provided

        Returns:
            Generated answer/summary
        """
        # Use provided vector store or instance default
        store = vector_store or self.vector_store
        if store is None:
            return "No vector store available for search"

        # Retrieve relevant documents
        try:
            results = store.query(query, top_k=top_k)
        except Exception as e:
            return f"Error during search: {e}"

        if not results:
            return "No relevant documents found for the query"

        # Combine document content
        context = "\n\n".join(
            [
                f"Document {i + 1}: {result['content']}"
                for i, result in enumerate(results)
            ]
        )

        # Generate answer using LLM
        try:
            # Format prompt
            prompt_text = self.prompt_template.format(context=context, question=query)

            # Generate response
            response = self.llm.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant that answers questions based on provided context."
                    ),
                    HumanMessage(content=prompt_text),
                ]
            )

            return response.content

        except Exception as e:
            return f"Error generating answer: {e}"
