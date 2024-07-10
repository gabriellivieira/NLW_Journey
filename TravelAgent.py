import os
from langchain_openai import ChatOpenAI
from langchain.agents.initialize import initialize_agent #Inicializa o agente
from langchain_community.agent_toolkits.load_tools import load_tools #Carrega as ferramentas do agente


llm = ChatOpenAI(model="gpt-3.5-turbo")

tools = load_tools(['ddg-search', 'wikipedia'], llm=llm)

#print(tools[1].name, tools[1].description)

agent = initialize_agent(
    tools,
    llm,
    agent= "zero-shot-react-description",
    verbose = True
)

#print(agent.agent.llm_chain.prompt.template)

query = """
Vou viajar para Londres em agosto de 2024. quero que faça um roteiro de viagem para mim com eventos que irão ocorrer na data da viagem e com o preço médio da passagem.
"""

agent.run(query)