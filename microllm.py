import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
import pandas as pd
import re
import json
import os
import pickle


class CommandParser:
    def __init__(self, model_path=None):
        """Initialize the command parser.

        Args:
            model_path: Optional path to load an existing model
        """
        self.label_mapping = {
            'create_user': 0,
            'delete_user': 1,
            'activate': 2,
            'deactivate': 3
        }
        self.intent_mapping = {v: k for k, v in self.label_mapping.items()}

        # Initialize or load model and tokenizer
        if model_path and os.path.exists(f"{model_path}/model.h5") and os.path.exists(f"{model_path}/tokenizer.pickle"):
            self.load(model_path)
            self.is_trained = True
        else:
            self.tokenizer = None
            self.model = None
            self.is_trained = False

    def generate_training_data(self, n_samples=500):
        """Generate synthetic training data."""
        # Define command templates and entities
        create_templates = [
            "create user {entity}",
            "add new user {entity}",
            "make user for {entity}",
            "create account for {entity}",
            "add user {entity}",
            "register new user {entity}",
            "set up user account for {entity}",
            "create profile for {entity}",
            "establish new user {entity}",
            "add user {entity} to system",
            "create user with name {entity}"
        ]

        delete_templates = [
            "delete user {entity}",
            "remove user account {entity}",
            "erase user profile {entity}",
            "delete account for {entity}",
            "remove user {entity}",
            "deactivate user {entity}",
            "eliminate user profile for {entity}",
            "remove access for user {entity}",
            "delete the account of {entity}",
            "purge user {entity} from system"
        ]

        activate_templates = [
            "activate feature {entity}",
            "enable {entity}",
            "turn on {entity}",
            "activate account for {entity}",
            "enable {entity} subscription",
            "activate system {entity}",
            "turn on automatic {entity}",
            "enable {entity} access",
            "activate new module {entity}",
            "enable {entity} access for developers"
        ]

        deactivate_templates = [
            "deactivate {entity} notifications",
            "disable {entity} account",
            "turn off {entity}",
            "deactivate old {entity}",
            "disable background {entity}",
            "turn off {entity} tracking",
            "deactivate unused {entity}",
            "disable system {entity}",
            "turn off {entity} mode",
            "deactivate temporary {entity}"
        ]

        # Define possible entities
        user_names = ["john", "sarah", "david", "lisa", "michael", "emma", "tom", "jacob",
                      "alex", "olivia", "sam", "rachel", "dan", "jessica", "george", "emily",
                      "chris", "amanda", "robert", "sophia", "william", "jennifer", "carlos",
                      "hannah", "maria", "johnson", "smith", "rodriguez", "lee", "nguyen",
                      "patel", "garcia", "wilson", "anderson", "thomas", "taylor", "moore",
                      "jackson", "martin", "kim", "singh", "kumar", "lewis", "robinson", "anna"]

        feature_names = ["dark mode", "notifications", "two factor authentication", "backup",
                         "premium", "system backup", "updates", "guest access", "statistics",
                         "api access", "email", "location", "auto-renewal", "server",
                         "processes", "debug mode", "modules", "alerts", "analytics",
                         "sharing", "sync", "cloud storage", "offline mode", "auto-save",
                         "encryption", "password protection", "audit logging", "remote access",
                         "ai suggestions", "voice commands", "device pairing", "screen sharing"]

        # Generate commands
        commands = []
        labels = []
        entities = []

        # Calculate samples per category
        samples_per_category = n_samples // 4

        # Create user commands
        for _ in range(samples_per_category):
            template = np.random.choice(create_templates)
            entity = np.random.choice(user_names)
            command = template.format(entity=entity)
            commands.append(command)
            labels.append("create_user")
            entities.append({"name": entity})

        # Delete user commands
        for _ in range(samples_per_category):
            template = np.random.choice(delete_templates)
            entity = np.random.choice(user_names)
            command = template.format(entity=entity)
            commands.append(command)
            labels.append("delete_user")
            entities.append({"name": entity})

        # Activate commands
        for _ in range(samples_per_category):
            template = np.random.choice(activate_templates)
            entity = np.random.choice(feature_names)
            command = template.format(entity=entity)
            commands.append(command)
            labels.append("activate")
            entities.append({"feature": entity})

        # Deactivate commands
        for _ in range(samples_per_category):
            template = np.random.choice(deactivate_templates)
            entity = np.random.choice(feature_names)
            command = template.format(entity=entity)
            commands.append(command)
            labels.append("deactivate")
            entities.append({"feature": entity})

        # Convert to DataFrame and shuffle
        df = pd.DataFrame({'command': commands, 'intent': labels, 'entity': entities})
        df = df.sample(frac=1).reset_index(drop=True)

        return df

    def train(self, df=None, epochs=15, save_path=None):
        """Train the model on provided or generated data."""
        # Generate data if not provided
        if df is None:
            df = self.generate_training_data(500)

        # Prepare text data
        self.tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(df['command'])
        word_index = self.tokenizer.word_index

        # Convert text to sequences
        sequences = self.tokenizer.texts_to_sequences(df['command'])
        padded_sequences = pad_sequences(sequences, maxlen=20, padding='post')

        # Convert labels to numerical form
        numerical_labels = df['intent'].map(self.label_mapping).values

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            padded_sequences, numerical_labels, test_size=0.2, random_state=42
        )

        # Create a neural network model
        self.model = Sequential([
            Embedding(input_dim=len(word_index) + 1, output_dim=32, input_length=20),
            Bidirectional(LSTM(64, return_sequences=True)),
            Bidirectional(LSTM(32)),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(len(self.label_mapping), activation='softmax')
        ])

        # Compile the model
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_test, y_test),
            batch_size=32,
            verbose=1
        )

        # Evaluate the model
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"\nTest accuracy: {accuracy:.4f}")

        self.is_trained = True

        # Save model if path provided
        if save_path:
            self.save(save_path)

        return history

    def save(self, path):
        """Save the model and tokenizer."""
        if not os.path.exists(path):
            os.makedirs(path)

        # Save model
        self.model.save(f"{path}/model.h5")

        # Save tokenizer
        with open(f"{path}/tokenizer.pickle", 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        """Load a saved model and tokenizer."""
        # Load model
        self.model = load_model(f"{path}/model.h5")

        # Load tokenizer
        with open(f"{path}/tokenizer.pickle", 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def extract_entity(self, command, intent):
        """Extract entity details from a command based on intent."""
        entity_data = {}

        if intent in ['create_user', 'delete_user']:
            # Look for name patterns
            name_patterns = [
                r"(?:user|for|of|name)\s+([a-z\s]+?)(?:\s+to|\s+from|$)",  # "user X", "for X", "name X"
                r"with\s+name\s+([a-z\s]+?)(?:\s+to|\s+from|$)",  # "with name X"
            ]

            for pattern in name_patterns:
                match = re.search(pattern, command)
                if match:
                    entity_data["name"] = match.group(1).strip()
                    break

            # If no match found, try to extract the last word
            if not entity_data and "user" in command:
                words = command.split()
                user_idx = words.index("user")
                if user_idx < len(words) - 1:
                    entity_data["name"] = words[user_idx + 1]

        elif intent in ['activate', 'deactivate']:
            # Look for feature patterns
            feature_patterns = [
                r"(?:activate|deactivate|enable|disable|turn\s+on|turn\s+off)\s+([a-z\s]+?)(?:\s+for|\s+to|\s+from|$)",
                r"(?:feature|system|module)\s+([a-z\s]+?)(?:\s+for|\s+to|\s+from|$)"
            ]

            for pattern in feature_patterns:
                match = re.search(pattern, command)
                if match:
                    entity_data["feature"] = match.group(1).strip()
                    break

            # If no match found, extract words after the action verb
            if not entity_data:
                action_words = ['activate', 'deactivate', 'enable', 'disable', 'turn']
                for word in action_words:
                    if word in command:
                        idx = command.find(word) + len(word)
                        remainder = command[idx:].strip()
                        if remainder:
                            entity_data["feature"] = remainder
                            break

        return entity_data

    def parse_command(self, command_text):
        """Parse a command into a structured JSON format."""
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first or load a saved model.")

        # Preprocess the command
        sequence = self.tokenizer.texts_to_sequences([command_text])
        padded = pad_sequences(sequence, maxlen=20, padding='post')

        # Get prediction
        prediction = self.model.predict(padded)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]

        # Map back to intent
        predicted_intent = self.intent_mapping[predicted_class]

        # Extract entity
        entity_data = self.extract_entity(command_text.lower(), predicted_intent)

        # Format JSON response
        result = {
            "what": predicted_intent,
            "data": entity_data,
            "confidence": float(confidence)
        }

        return result

    def interactive_mode(self):
        """Run an interactive command parsing session."""
        if not self.is_trained:
            print("Model is not trained. Training now...")
            self.train()

        print("\n===== Command Parser Interactive Mode =====")
        print("Type 'exit' or 'quit' to end the session")
        print("Type 'json' to toggle JSON output mode")
        print("=========================================\n")

        json_mode = True

        while True:
            user_input = input("\nEnter a command: ").strip()

            if user_input.lower() in ['exit', 'quit']:
                print("Exiting interactive mode. Goodbye!")
                break

            if user_input.lower() == 'json':
                json_mode = not json_mode
                print(f"JSON output mode: {'ON' if json_mode else 'OFF'}")
                continue

            try:
                result = self.parse_command(user_input)

                if json_mode:
                    print(json.dumps(result, indent=2))
                else:
                    intent = result['what']
                    data = result['data']
                    confidence = result['confidence']

                    print(f"Intent: {intent} (Confidence: {confidence:.2f})")
                    print(f"Data: {data}")

            except Exception as e:
                print(f"Error processing command: {e}")


# Main execution
if __name__ == "__main__":
    parser = CommandParser()

    # Check if a saved model exists
    if os.path.exists("model/model.h5"):
        print("Loading existing model...")
        parser.load("model")
    else:
        print("Training new model...")
        parser.train(save_path="model")

    parser.interactive_mode()