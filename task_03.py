import nltk
import spacy
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# Download all required NLTK data
def setup_nltk():
    resources = [
        'punkt',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'punkt_tab'
    ]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {resource}: {str(e)}")
            
    # Verify punkt tokenizer specifically
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading additional required resources...")
        nltk.download('punkt')

class IntentRecognizer:
    def __init__(self):
        self.intents = {
            'greeting': ['hello', 'hi', 'hey', 'howdy', 'greetings'],
            'farewell': ['bye', 'goodbye', 'see you', 'farewell'],
            'thanks': ['thank', 'thanks', 'appreciate'],
            'help': ['help', 'support', 'assist'],
            'about': ['who are you', 'what are you', 'tell me about yourself']
        }
    
    def get_intent(self, text):
        # Word matching 
        text_lower = text.lower()
        words = text_lower.split()
        
        for intent, patterns in self.intents.items():
            if any(pattern in words or pattern in text_lower for pattern in patterns):
                return intent
        return 'unknown'

class EntityExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            self.nlp = spacy.load('en_core_web_sm')
        
    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = defaultdict(list)
        
        for ent in doc.ents:
            entities[ent.label_].append(ent.text)
            
        return dict(entities)

class ResponseGenerator:
    def __init__(self):
        self.responses = {
            'greeting': [
                "Hello! How can I help you today?",
                "Hi there! What's on your mind?",
                "Greetings! How may I assist you?"
            ],
            'farewell': [
                "Goodbye! Have a great day!",
                "See you later! Take care!",
                "Farewell! Feel free to return if you need more help!"
            ],
            'thanks': [
                "You're welcome!",
                "Glad I could help!",
                "It's my pleasure!"
            ],
            'help': [
                "I can help you with various topics. What specific assistance do you need?",
                "I'm here to help! Please let me know what you're looking for.",
                "Sure, I'd be happy to help. What do you need?"
            ],
            'about': [
                "I'm a chatbot designed to help users with their queries.",
                "I'm an AI assistant here to help you with various tasks and questions.",
                "I'm a virtual assistant powered by NLP, ready to help you!"
            ],
            'unknown': [
                "I'm not sure I understand. Could you please rephrase that?",
                "I didn't quite catch that. Can you explain differently?",
                "I'm still learning! Could you try asking in another way?"
            ]
        }
        
    def generate_response(self, intent, entities=None):
        response = random.choice(self.responses[intent])
        
        if entities:
            if 'PERSON' in entities:
                response += f" I notice you mentioned {entities['PERSON'][0]}."
            if 'ORG' in entities:
                response += f" Regarding {entities['ORG'][0]}..."
            if 'DATE' in entities:
                response += f" About the date {entities['DATE'][0]}..."
                
        return response

class Chatbot:
    def __init__(self):
        # NLTK resources
        setup_nltk()
        
        self.intent_recognizer = IntentRecognizer()
        self.entity_extractor = EntityExtractor()
        self.response_generator = ResponseGenerator()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        # Avoid tokenization issues
        words = text.lower().split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return ' '.join(words)
        
    def chat(self, user_input):
        try:
            # Preprocess input
            processed_input = self.preprocess_text(user_input)
            
            # Recognize intent
            intent = self.intent_recognizer.get_intent(processed_input)
            
            # Extract entities
            entities = self.entity_extractor.extract_entities(user_input)
            
            # Generate response
            response = self.response_generator.generate_response(intent, entities)
            
            return response
        except Exception as e:
            print(f"Error processing input: {str(e)}")
            return "I encountered an error. Could you try saying that differently?"

def main():
    print("Initializing chatbot...")
    print("Chatbot: Hello! I'm ready to chat. Type 'bye' to exit.")
    chatbot = Chatbot()
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['bye', 'goodbye', 'exit']:
                print("Chatbot: Goodbye! Have a great day!")
                break
                
            response = chatbot.chat(user_input)
            print("Chatbot:", response)
        except KeyboardInterrupt:
            print("\nChatbot: Goodbye! Have a great day!")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Chatbot: I'm having trouble processing that. Let's continue our conversation.")

if __name__ == "__main__":
    main()