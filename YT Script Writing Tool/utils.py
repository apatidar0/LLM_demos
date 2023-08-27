from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import DuckDuckGoSearchRun
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain import FewShotPromptTemplate



# Function to generate video script
def generate_script(prompt,video_length,creativity,api_key):

    # Example template for Fewshot learning
    examples = [
    {
        "topic": "Machine learning",
        "answer": "Unlocking Tomorrow: Explore the Fascinating Frontiers of Machine Learning!"
    }, {
        "topic": "sales crm",
        "answer": "Supercharge Your Sales with CRM Mastery: Unleashing Success and Growth!"
    },
    {
        "topic": "DIY Fashion: Upcycling and Customizing Clothes",
        "answer": "Elevate Your Style: Master the Art of DIY Fashion, Upcycling, and Personalized Attire!"
    },
    {
        "topic": "Fascinating Facts About Automatic using AI",
        "answer": "Revealing the Future: Amazing Insights into AI-Powered Automation and its Unbelievable Capabilities!"
    }
    ]

    example_template = """
    Topic: {topic}
    Title: {answer}
    """

    example_prompt = PromptTemplate(
        input_variables=["topic", "answer"],
        template=example_template
    )

    prefix = """<<SYS>>\n Craft titles that evoke curiosity, highlight the excitement of exploration, and make viewers eager to look into it.
    Aim for a mix of intrigue, excitement, and value to maximize viewer engagement:
    Here are some examples:
    """

    suffix = """<</SYS>>\n\n [INST]
    Generate attention-grabbing YouTube titles for the given topic in a single line.
    Topic: {userInput}
    Title: [/INST]"""


    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["userInput"],
        example_separator="\n\n"
    )
    
    
    
    # Template for generating 'Video Script' using search engine

    sc_template = """<<SYS>>\nChannel your scriptwriting expertise to craft a compelling YouTube video script. Transform the given text into an engaging narration, weaving in the topic seamlessly, captivating your audience, and concluding with a compelling call to action. Your role is crucial in ensuring viewer engagement through well-structured, captivating content.\n<</SYS>>\n\n
    [INST] Create a script for a YouTube video based on this title for me. TITLE: {title} of duration: {duration} minutes using this search data {DuckDuckGo_Search} [/INST]"""

    
    script_template = PromptTemplate(
        input_variables = ['title', 'DuckDuckGo_Search','duration'],
        template = sc_template)

    #Setting up OpenAI LLM
    # llm = OpenAI(temperature=creativity,openai_api_key=api_key,
    #         model_name='gpt-3.5-turbo') 

    # Setting up Opensource LLama2 7B chat ggml (cpu) model
    llm = CTransformers(model='C:/Users/Dell/Desktop/demos/Email Generator App - Source Code/models/llama-2-7b-chat.ggmlv3.q8_0.bin',     #https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
                    model_type='llama',
                    config={'max_new_tokens': 512,
                            'temperature': 0})

    
    #Creating chain for 'Title' & 'Video Script'
    title_chain = LLMChain(llm=llm, prompt=few_shot_prompt_template, verbose=True)
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True)

    
    # https://python.langchain.com/docs/modules/agents/tools/integrations/ddg
    search = DuckDuckGoSearchRun()

    # Executing the chains we created for 'Title'
    title = title_chain.run(prompt)

    # Executing the chains we created for 'Video Script' by taking help of search engine 'DuckDuckGo'
    search_result = search.run(prompt) 
    script = script_chain.run(title=title, DuckDuckGo_Search=search_result,duration=video_length)

    # Returning the output
    return search_result,title,script