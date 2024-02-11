import os
from crewai import Task, Crew, Agent, Process
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
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
#topic = input("Please enter the topic: ")
topic = """Create an elegant and professional resume example in markedown format for a computer science major 
  student who also competes in NCAA wrestling, seeking a summer internship in the computer science field. """

tip = "If you do your BEST WORK, I'll give you a $10,000 commission!"

# Loading Human Tools
human_tools = load_tools(["human"])

# Creating a senior researcher agent
researcher = Agent(
  role='Senior Researcher',
  goal=f'Uncover effective and successful job seeking tactics around {topic}',
  verbose=True,
  backstory="""You are a Senior Research Analyst at a leading job recruiting agency for top tech companies.
  Your expertise lies in identifying emerging trends and templates in resume development and
  job applications. You have a knack for dissecting complex data and presenting
  actionable insights.""",
  allow_delegation=False,
  # Passing human tools to the agent
  tools=[search_tool_duck],
  llm=llm_model_select
)

# Creating a writer agent
writer = Agent(
  role='Writer',
  goal=f'Craft compelling, elegant and professional resumes around {topic}',
  verbose=True,
  backstory="""You are a renowned tech resume writer for top university students, known for your insightful
  and engaging articles on resume development and job seeking skills. With a deep understanding of
  the job recruiting industry, you transform complex concepts into compelling narratives.""",
  allow_delegation=True,
  llm=llm_model_select
)

# Creating a writer agent
director = Agent(
  role='Executive Director',
  goal="""Oversee the work done by your team to make sure it's the best
	possible and aligned with the product's goals, review, approve,
	ask clarifying question or delegate follow up work if necessary to make decisions""",
  backstory="""You're the Executive Director of leading job recruiting firm specialized in finding top college talent for elite tech companies. 
  You're working on a new student customer, trying to make sure your team is crafting the best possible
  content for the customer. You are also responsible for ensuring that the project meets the client's 
  requirements and that the final deliverable is of the highest quality.""",
  verbose=True,
  allow_delegation=True,
  llm=llm_model_select
)

# Research task for identifying AI trends
research_task = Task(
  description=f"""Conduct a comprehensive analysis of the {topic}.
  Collect and summarize recent top resume templates for university students in computer science major
  to help prepare for a university job fair.
  Pay special attention to any significant templates that include NCAA athelete in accomplishments.
  
  {tip}
  
  Your final answer MUST be a report that includes a comprehensive summary of the most impactful resume templates for 
  university students going to a job fair.
  Your final answer MUST be a full analysis report.
  """,
  expected_output=f'Provide a comprehensive list of key takeways and uncommon insights based on the {topic}.',
  agent=researcher  
)

# Writing task based on research findings
write_task = Task(
  description=f"""Using the insights from the researcher's report, 
  Create an elegant and professional resume example in markedown format for a computer science major 
  student who also competes in NCAA wrestling, seeking a summer internship in the computer science field. 
  
  {tip}
  
  The template should have a clean, modern design, with sections for educational background, technical skills, 
  wrestling achievements, work experience, and projects. The education section should highlight their computer 
  science major, relevant coursework, and GPA. The technical skills section should detail programming 
  languages (e.g., Python, Java, C++), tools (e.g., Git, Docker), and technologies 
  (e.g., Machine Learning, Web Development) they are proficient in. The wrestling achievements 
  should not only list awards and recognitions but also emphasize teamwork, discipline, and resilience. 
  The work experience and projects sections should showcase their ability to apply computer science 
  knowledge in practical settings, highlighting any previous internships, part-time jobs, or significant 
  classroom projects. Include placeholders for personal information, a professional summary that ties 
  together their dual passion for computer science and wrestling, and a section for extracurricular 
  activities that further demonstrate their leadership and teamwork skills. The design should incorporate 
  subtle graphics that suggest technology and athleticism without overwhelming the content, ensuring the 
  focus remains on the student's qualifications and achievements.
    """,
  expected_output=f'A full resume example in markdown format.',
  agent=writer
)

# Director task based on improving the resume example from the writer's finding
director_task = Task(
  description=f"""Using the resume example from the writer, leveraging your experience as a top tech industry recruiter make it 10X better.
  {tip} """,
  expected_output=f'A world class professional resume example in markdown format.',
  agent=writer
)

# Forming the resume-focused crew
crew = Crew(
  agents=[researcher, writer, director],
  tasks=[research_task, write_task, director_task],
  process=Process.sequential,  # Sequential task execution
  verbose=True,
  full_output=True
)

# Starting the task execution process
result = crew.kickoff()
print("######################")
print(result)
