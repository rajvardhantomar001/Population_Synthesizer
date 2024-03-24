import os
from settings import OPENAI_API_KEY #Contains the API details
import json
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.pydantic_v1 import BaseModel
from langchain_experimental.tabular_synthetic_data.openai import (
    create_openai_data_generator,
    OPENAI_TEMPLATE,
)
from langchain_experimental.tabular_synthetic_data.prompts import SYNTHETIC_FEW_SHOT_SUFFIX
from langchain_experimental.tabular_synthetic_data.prompts import SYNTHETIC_FEW_SHOT_PREFIX

# Parameters for the Road Safety Dataset
class RoadSafetyData(BaseModel):
    Vehicle_ID: str
    Vehicle_Type: str
    Driver_Age: int
    Road_Type: str
    Weather_Conditions: str
    Accident_Type: str

# Examples of road safety data
examples = [
    {
        "example": """Vehicle_ID: UP14AD7811, Vehicle_Type: Sedan, Driver_Age: 35, Road_Type: Highway, Weather_Conditions: Clear, Accident_Type: Collision"""
    },
    {
        "example": """Vehicle_ID: DL12DA9833, Vehicle_Type: SUV, Driver_Age: 28, Road_Type: Urban, Weather_Conditions: Rainy, Accident_Type: Rollover"""
    },
    {
        "example": """Vehicle_ID: AP22AA9900, Vehicle_Type: Truck, Driver_Age: 45, Road_Type: Rural, Weather_Conditions: Foggy, Accident_Type: Rear-end"""
    },
]

# Template for generating synthetic data using OpenAI
OPENAI_TEMPLATE = PromptTemplate(input_variables=["example"], template="{example}")

# Few-shot prompt template for synthetic data generation
prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    examples=examples,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    input_variables=["subject", "extra"],
    example_prompt=OPENAI_TEMPLATE,
)

# Set OpenAI API key
os.environ['OPENAI_API_KEY']= OPENAI_API_KEY

# Create synthetic data generator
synthetic_data_generator = create_openai_data_generator(
    output_schema=RoadSafetyData,
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125"),
    prompt=prompt_template,
)

# Generate synthetic road safety data
synthetic_results = synthetic_data_generator.generate(
    subject="Road_Safety_Data",
    extra="generate synthetic road safety data.",
    runs=3,  # Number of synthetic data points to generate
)

# File path for saving synthetic data
txt_file_path = 'road_safety_synthetic_data.txt'

# Write the synthetic results to the text file
with open(txt_file_path, 'w') as txt_file:
    # Write each synthetic data point as a JSON object in the text file
    for road_safety_data in synthetic_results:
        json.dump(road_safety_data.dict(), txt_file)
        txt_file.write('\n')  # Add a newline after each JSON object

print(f'Synthetic road safety data has been saved to {txt_file_path}')