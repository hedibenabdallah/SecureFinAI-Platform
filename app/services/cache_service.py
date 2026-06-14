import uuid

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION_NAME = "proxy_cache"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536


class CacheService:
    def __init__(self, qdrant_url: str, openai_api_key: str, threshold: float) -> None:
        self._client = AsyncQdrantClient(url=qdrant_url)
        self._openai = AsyncOpenAI(api_key=openai_api_key)
        self._threshold = threshold

    async def setup(self) -> None:
        exists = await self._client.collection_exists(COLLECTION_NAME)
        if not exists:
            await self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    async def lookup(self, prompt: str) -> str | None:
        vector = await self._embed(prompt)
        results = await self._client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=1,
        )
        if not results:
            return None
        top = results[0]
        if top.score < self._threshold:
            return None
        payload = top.payload or {}
        response = payload.get("response")
        if isinstance(response, str):
            return response
        return None

    async def store(self, prompt: str, response: str) -> None:
        vector = await self._embed(prompt)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"prompt": prompt, "response": response},
        )
        await self._client.upsert(collection_name=COLLECTION_NAME, points=[point])

    async def _embed(self, text: str) -> list[float]:
        result = await self._openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return result.data[0].embedding
