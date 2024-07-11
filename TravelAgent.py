import os
import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_react_agent, AgentExecutor #Inicializa o agente

from langchain_community.agent_toolkits.load_tools import load_tools #Carrega as ferramentas do agente
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
import bs4  #para interpretar o html do site

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence


OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

llm = ChatOpenAI(model="gpt-3.5-turbo")

# Esse query foi usada para testar a lambda na AWS.
#query = """
#Vou viajar para Londres em agosto de 2024. 
#Quero que faça um roteiro de viagem para mim com eventos que irão ocorrer na data da viagem e com o preço médio da passagem.
#"""

#Função que chama o agente de pesquisa
def researchAgent(query, llm):
    tools = load_tools(['ddg-search', 'wikipedia'], llm=llm)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, prompt=prompt)
    webContext = agent_executor.invoke({"input":query})
    return webContext['output']

#Função que irá carregar as informações dentro de um banco vetorial para que o agente de viagem use como base de informação.
def loadData():
    loader = WebBaseLoader(
    web_paths= ("https://www.dicasdeviagem.com/inglaterra/",),
    bs_kwargs= dict(parse_only=bs4.SoupStrainer(class_=("postcontentwrap", "pagetitleloading background-imaged loading-dark"))), #são os parametros do bs4 que serão nomeados para converter e chamar o loader
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) #separa os conteudo da página em caracteres para então fazer o embedded
    splits = text_splitter.split_documents(docs) #Executar o metodo com os docs que foram carregados
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings()) #Salvando em um banco vetorial
    retriever = vectorstore.as_retriever()
    return retriever

#Função para utilizar os documentos mais relevantes capturados e preparados na função loadData 
def getRelevantDocs(query):
    retriever = loadData()
    relevant_documents = retriever.invoke(query)
    #print(relevant_documents)
    return relevant_documents

#Função para contruir um supervidor que vai consumir os dados da internet gerado pelo agent (researchAgent) e os docuemntos relevantes (getRelevantDocs)
def supervisorAgent(query, llm, webContext, relevant_documents):
    prompt_template = """
    Você é um agente de uma agencia de viagens. Sua resposta final deverá ser um roteiro de viagem completo e detalhado.
    Utilize o contexto de eventos e preços de passagens, o input do usuário e também os documentos relevantes para elaborar o roteiro.
    Contexto: {webContext}
    Documento relevante: {relevant_documents}
    Usuáio: {query}
    Assistente:
    """
    prompt = PromptTemplate(
        input_variables = ['webContext', 'relevant_documents', 'query'],
        template = prompt_template
    )

    sequence = RunnableSequence(prompt | llm)
    response = sequence.invoke({"webContext": webContext, "relevant_documents": relevant_documents, "query": query })
    return response

#Função que faz o encadeamento das chamadas
def getResponse(query, llm):
    webContext = researchAgent(query, llm)
    relevant_documents = getRelevantDocs(query)
    response = supervisorAgent(query, llm, webContext, relevant_documents)
    return response 

#print(getResponse(query, llm).content)

#
def lambda_handler(event, context):
    #query = event.get("question")
    body = json.loads(event.get('body', {}))
    body = json.loads(event.get('body', {}))
    query = body.get('question', 'Parametro question não fornecido')
    response = getResponse(query, llm).content
    return {
        "statusCode": 200,
        "headers": {
        "Content-Type": "application/json"
        },
        "body": json.dumps({
        "message": "Tarefa concluída com sucesso",
        "details": response
        }), 
    }