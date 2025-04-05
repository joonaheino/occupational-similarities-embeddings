import json
import time
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = OpenAI(api_key=OPENAI_API_KEY)

def load_json(filename):
    """Load and read a JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def save_json(data, filename):
    """Save data to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def get_gpt4_evaluation(breed, top_traits, bottom_traits):
    """
    Ask GPT-4 to evaluate and score traits for a specific breed.
    """
    prompt = f"""As a canine behavior expert, please evaluate how well each trait matches the {breed} breed.
    
For each trait, assign a score from 1-100 where:
- 100 means the trait perfectly describes the breed
- 1 means the trait is completely mismatched with the breed
- Scores should be well-distributed across the range

Please evaluate these traits that were identified as most similar:
{[t['characteristic'] for t in top_traits]}

And these traits that were identified as least similar:
{[t['characteristic'] for t in bottom_traits]}

Respond in this JSON format:
{
    "top_traits": [
        {"trait": "trait_name", "score": score, "explanation": "brief explanation"},
        ...
    ],
    "bottom_traits": [
        {"trait": "trait_name", "score": score, "explanation": "brief explanation"},
        ...
    ]
}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a canine behavior expert who understands dog breeds deeply."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting GPT-4 evaluation for {breed}: {e}")
        return None

def main():
    # Load the similarity data
    similarities = load_json('breed_characteristic_similarities.json')
    
    # Create dictionary for GPT-4 evaluations
    gpt4_evaluations = {}
    
    # Process each breed
    for breed in similarities["breed_to_characteristics"]:
        print(f"\nProcessing {breed}...")
        
        # Get top 10 and bottom 10 characteristics
        traits = similarities["breed_to_characteristics"][breed]
        top_10 = traits[:10]
        bottom_10 = traits[-10:]
        
        # Get GPT-4 evaluation
        evaluation = get_gpt4_evaluation(breed, top_10, bottom_10)
        
        if evaluation:
            gpt4_evaluations[breed] = evaluation
            # Save after each breed in case of interruption
            save_json(gpt4_evaluations, 'gpt4_trait_evaluations.json')
            print(f"Saved evaluation for {breed}")
        
        # Sleep to respect rate limits
        time.sleep(3)
    
    print("\nAll evaluations completed and saved to gpt4_trait_evaluations.json")
    
    # Print example evaluation
    sample_breed = list(gpt4_evaluations.keys())[0]
    print(f"\nExample evaluation for {sample_breed}:")
    print(json.dumps(gpt4_evaluations[sample_breed], indent=2))

if __name__ == "__main__":
    main() 