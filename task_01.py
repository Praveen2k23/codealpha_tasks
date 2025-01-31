import random

class HangmanGame:
    def __init__(self):
        # Word list with hints 
        self.word_dictionary = [
            ("python", "A popular programming language named after a snake"),
            ("programming", "The process of writing computer instructions"),
            ("computer", "An electronic device that processes data"),
            ("algorithm", "A step-by-step procedure for solving a problem"),
            ("database", "A structured collection of data"),
            ("network", "A system of interconnected computers"),
            ("software", "Programs and other operating information used by computers"),
            ("developer", "Someone who creates computer programs"),
            ("internet", "Global computer network providing information and communication"),
            ("keyboard", "Device used to input text into a computer")
        ]
        self.max_tries = 6
    
    def get_random_word(self):
        return random.choice(self.word_dictionary)
    
    def display_game_state(self, word, hint, guessed_letters, tries):
        # Display word with guessed letters
        display_word = ' '.join([letter if letter in guessed_letters else '_' for letter in word])
        print(f"\nWord: {display_word}")
        print(f"Hint: {hint}")
        print(f"Guessed letters: {' '.join(sorted(guessed_letters)) if guessed_letters else 'None'}")
        print(f"Remaining tries: {self.max_tries - tries}\n")
    
    def play(self):
        print("\nWelcome to Hangman!")
        print("Try to guess the word, one letter at a time. You have 6 incorrect guesses before you lose.")
        print("A hint is provided to help you. Good luck!\n")
        
        word, hint = self.get_random_word()
        guessed_letters = set()
        tries = 0
        
        while tries < self.max_tries:
            self.display_game_state(word, hint, guessed_letters, tries)
            
            guess = input("Guess a letter: ").lower()
            
            if len(guess) != 1 or not guess.isalpha():
                print("Invalid input! Please enter a single letter.")
                continue
            if guess in guessed_letters:
                print("You already guessed that letter!")
                continue
            
            guessed_letters.add(guess)
            
            if guess not in word:
                tries += 1
                print("Incorrect guess!")
                if tries == self.max_tries:
                    self.display_game_state(word, hint, guessed_letters, tries)
                    print(f"\nGame Over! The word was: {word}\n")
                    break
            
            if all(letter in guessed_letters for letter in word):
                self.display_game_state(word, hint, guessed_letters, tries)
                print("\nCongratulations! You won!\n")
                break
    
    def start_game(self):
        while True:
            self.play()
            play_again = input("Would you like to play again? (y/n): ").lower()
            if play_again != 'y':
                print("\nThanks for playing Hangman!")
                break

if __name__ == "__main__":
    game = HangmanGame()
    game.start_game()
