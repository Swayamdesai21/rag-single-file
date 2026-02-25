from app.rag_pipeline import build_rag_pipeline

def ask_question(query: str):
    rag = build_rag_pipeline()
    response = rag.invoke(query)

    print("\nAnswer:\n")
    print(response.content)


if __name__ == "__main__":
    user_query = input("Ask your question: ")
    ask_question(user_query)
