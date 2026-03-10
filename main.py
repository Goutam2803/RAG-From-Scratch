from no_rag_pipeline import ask_llm
from rag_pipeline import ask_with_rag

def run_demo():

    question = input("\nEnter your question: ")

    print("\n==============================")
    print("WITHOUT RAG")
    print("==============================")

    answer_no_rag = ask_llm(question)

    print("\nAnswer:")
    print(answer_no_rag)

    print("\n==============================")
    print("WITH RAG")
    print("==============================")

    answer_rag, context = ask_with_rag(question)

    print("\nRetrieved Context:")
    print(context)

    print("\nAnswer:")
    print(answer_rag)


if __name__ == "__main__":
    run_demo()