

class RAG:
    def __init__(self,embedding_function, model, chroma_db):
        self.embedding_function = embedding_function
        self.model = model
        self.chroma_db = chroma_db

    def retrieve(self, query):
        query_embedding = self.embedding_function.embed_image(query)
        retrieved_documents = self.chroma_db.retrieve(query_embedding)

        # TODO: implement 1.get outfits 2. pass it to the model

        return retrieved_documents

    def generate_response(self, retrieved_documents):

        response = self.model.generate_response(retrieved_documents)
        return response
    