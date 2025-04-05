import json
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize

def load_json(filename):
    """Load and read a JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def save_json(data, filename):
    """Save data to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def cosine_similarity(v1, v2):
    """
    Compute cosine similarity between two vectors.
    Now with explicit normalization like OpenAI's implementation.
    """
    # Normalize the vectors to unit length
    v1_norm = normalize(v1.reshape(1, -1))[0]
    v2_norm = normalize(v2.reshape(1, -1))[0]
    # Compute dot product of normalized vectors
    return np.dot(v1_norm, v2_norm)

def main():
    # Load the embeddings
    breeds_data = load_json('dog-breeds-with-embeddings.json')
    characteristics_data = load_json('breed-characteristics-with-embeddings.json')
    
    # Flatten breeds and characteristics into lists with their embeddings
    all_breeds = []
    for category in breeds_data.values():
        all_breeds.extend(category)
    
    all_characteristics = []
    for category in characteristics_data.values():
        all_characteristics.extend(category)
    
    print(f"Computing similarities between {len(all_breeds)} breeds and {len(all_characteristics)} characteristics...")
    
    # Create similarity mappings
    similarities = {
        "breed_to_characteristics": {},  # Each breed's similarity to all characteristics
        "characteristic_to_breeds": {}   # Each characteristic's similarity to all breeds
    }
    
    # Compute similarities: breeds -> characteristics
    for breed_entry in all_breeds:
        breed_name = breed_entry['breed']
        breed_embedding = np.array(breed_entry['embedding'])
        
        breed_similarities = []
        for char_entry in all_characteristics:
            char_name = char_entry['trait']
            char_embedding = np.array(char_entry['embedding'])
            
            similarity = cosine_similarity(breed_embedding, char_embedding)
            breed_similarities.append({
                "characteristic": char_name,
                "similarity": float(similarity)  # Convert to float for JSON serialization
            })
        
        # Sort by similarity in descending order
        breed_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        similarities["breed_to_characteristics"][breed_name] = breed_similarities
    
    # Compute similarities: characteristics -> breeds
    for char_entry in all_characteristics:
        char_name = char_entry['trait']
        char_embedding = np.array(char_entry['embedding'])
        
        char_similarities = []
        for breed_entry in all_breeds:
            breed_name = breed_entry['breed']
            breed_embedding = np.array(breed_entry['embedding'])
            
            similarity = cosine_similarity(char_embedding, breed_embedding)
            char_similarities.append({
                "breed": breed_name,
                "similarity": float(similarity)
            })
        
        # Sort by similarity in descending order
        char_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        similarities["characteristic_to_breeds"][char_name] = char_similarities
    
    # Save the results
    save_json(similarities, 'breed_characteristic_similarities.json')
    print("Similarities computed and saved to breed_characteristic_similarities.json")
    
    # Print some example statistics
    sample_breed = list(similarities["breed_to_characteristics"].keys())[0]
    sample_char = list(similarities["characteristic_to_breeds"].keys())[0]
    
    print("\nExample top 5 characteristics for", sample_breed)
    for sim in similarities["breed_to_characteristics"][sample_breed][:5]:
        print(f"{sim['characteristic']}: {sim['similarity']:.3f}")
    
    print("\nExample top 5 breeds for", sample_char)
    for sim in similarities["characteristic_to_breeds"][sample_char][:5]:
        print(f"{sim['breed']}: {sim['similarity']:.3f}")

if __name__ == "__main__":
    main() 