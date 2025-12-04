#pip install langchain-openai httpx

from langchain_openai import ChatOpenAI
import os
import httpx

# Use your real hackathon key here
API_KEY = os.getenv("GENAILAB_API_KEY", "sk-QibkR-xrOxD2AsMgdaL0Pg")

client = httpx.Client(verify=False)

llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key=API_KEY,
    http_client=client,
    temperature=0.2,
)

# Simple test
response = llm.invoke("Hi")
print(response)