import os
from crewai import Task, Crew, Agent, Process
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key here
openai_api_key = os.environ.get("OPENAI_API_KEY")
llm_openai = ChatOpenAI(model_name="gpt-4-0125-preview", api_key=openai_api_key, verbose=True) 

# Create a DuckDuckGo search tool
search_tool_duck = DuckDuckGoSearchRun()

# Create an Ollama model
llm_openhermes = Ollama(model="openhermes")
llm_mistral = Ollama(model="mistral")

#llm_model_select = llm_openhermes
print("Welcome!")

#Define available models
models = {
    "openhermes": Ollama(model="openhermes"),
    "mistral": Ollama(model="mistral"),
    "openai": ChatOpenAI(model_name="gpt-4-0125-preview", api_key=os.environ.get("OPENAI_API_KEY"), verbose=True)
}

#Prompt user for model selection
model_choice = input("Select the model (openhermes, mistral): ").lower().strip()

# Set llm_model_select based on user input
if model_choice in models:
    llm_model_select = models[model_choice]
else:
    print("Thank you for using AI Chat!")
    exit()
    
# Ask the user to input the topic
topic = input("Please enter the topic: ")

# Creating a senior researcher agent
researcher = Agent(
  role='Senior Researcher',
  goal=f'Uncover groundbreaking technologies around {topic}',
  verbose=True,
  backstory="""You are a Senior Research Analyst at a leading tech think tank.
  Your expertise lies in identifying emerging trends and technologies in tech, AI and
  data science. You have a knack for dissecting complex data and presenting
  actionable insights.""",
  allow_delegation=False,
  # Passing human tools to the agent
  tools=[search_tool_duck],
  llm=llm_model_select
)

# Creating a writer agent
writer = Agent(
  role='Writer',
  goal=f'Craft compelling tech stories around {topic}',
  verbose=True,
  backstory="""You are a renowned Tech Content Strategist, known for your insightful
  and engaging articles on technology and innovation. With a deep understanding of
  the tech industry, you transform complex concepts into compelling narratives.""",
  allow_delegation=True,
  llm=llm_model_select
)

# Research task for identifying AI trends
research_task = Task(
  description=f"""Conduct a comprehensive analysis of the {topic}.
  Identify key trends, breakthrough technologies, and potential industry impacts.
  Compile your findings in a detailed report.
  Your final answer MUST be a full analysis report.
  """,
  expected_output=f'Provide a comprehensive list of key takeways and uncommon insights based on the {topic}.',
  agent=researcher  
)

# Writing task based on research findings
write_task = Task(
  description=f"""Using the insights from the researcher's report, 
  develop an engaging blog post that highlights the most significant insights.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Focus on the latest trends and how it's impacting the industry.
  This article should be easy to understand, engaging and positive.
  Your final answer MUST be the full blog post of at least 3 paragraphs.
    """,
  expected_output=f'A 4 paragraph article include bulleted list of key take aways and uncommon insights from researcher.',
  agent=writer
)

# Forming the tech-focused crew
crew = Crew(
  agents=[researcher, writer],
  tasks=[research_task, write_task],
  process=Process.sequential,  # Sequential task execution
  verbose=2
)

# Starting the task execution process
result = crew.kickoff()
print("######################")
print(result)
