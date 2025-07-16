import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from roadmap_tool import get_career_roadmap

# Load environment variables
load_dotenv()
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
# Initialize the agent with the model
model = OpenAIChatCompletionsModel(model="gemini-2.0-flash" , openai_client=client)
config = RunConfig(model=model, tracing_disabled=True)

career_agent = Agent(
    name="Career Roadmap Agent",
    instructions="An agent that provides career roadmaps for various fields.",
    model=model,
)

skill_agent = Agent(
    name="Skill Development Agent",  
    instructions="An agent that provides skill development resources and advice.",
    model=model,
    tools=[get_career_roadmap]  
)

job_agent = Agent(
    name="Job Search Agent",
    instructions="An agent that helps with job search strategies and resources.",
    model=model,
)

def main():
    print("Welcome to the Career Mentor Agent!")
    interest = input("What field are you interested in? (e.g., software development, data science): ")

    result1 = Runner.run_sync(career_agent, interest, run_config=config)
    field = result1.final_output.strip()
    print("\n Suggested Career : ", field)

    result2 = Runner.run_sync(skill_agent, interest, run_config=config)
    print("\n Reguired Skills for interest:", result2.final_output)

    result3 = Runner.run_sync(job_agent, interest, run_config=config)
    print("\n job Search Strategies for interest:", result3.final_output)

if __name__ == "__main__":
    main()








