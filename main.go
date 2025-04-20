package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"regexp"
	"strings"
)

// CommandResult represents the structured JSON output for a parsed command
type CommandResult struct {
	What       string                 `json:"what"`
	Data       map[string]interface{} `json:"data"`
	Confidence float64                `json:"confidence,omitempty"`
	Error      string                 `json:"error,omitempty"`
}

// Pattern contains a regex pattern with its associated weight for confidence calculation
type Pattern struct {
	Pattern *regexp.Regexp
	Weight  float64
}

// IntentPattern represents patterns for a specific intent
type IntentPattern struct {
	Patterns          []Pattern
	EntityExtractors  []Pattern
	EntityFieldName   string
	AdditionalParsers []func(string, map[string]interface{})
}

// CommandParser handles parsing text commands into structured data
type CommandParser struct {
	intentPatterns map[string]IntentPattern
	vocabulary     map[string]int
}

// NewCommandParser creates a new command parser with predefined patterns
func NewCommandParser() *CommandParser {
	cp := &CommandParser{
		intentPatterns: make(map[string]IntentPattern),
		vocabulary:     make(map[string]int),
	}

	// Build vocabulary (simplified version of what word embeddings do)
	commonWords := []string{
		"create", "add", "make", "user", "delete", "remove", "activate",
		"deactivate", "enable", "disable", "turn", "on", "off", "account",
		"profile", "new", "feature", "system", "for", "with", "name",
	}
	for i, word := range commonWords {
		cp.vocabulary[word] = i
	}

	// Create user patterns
	cp.intentPatterns["create_user"] = IntentPattern{
		Patterns: []Pattern{
			{regexp.MustCompile(`(?i)create\s+user`), 1.0},
			{regexp.MustCompile(`(?i)add\s+(?:new\s+)?user`), 0.9},
			{regexp.MustCompile(`(?i)make\s+user`), 0.8},
			{regexp.MustCompile(`(?i)register\s+(?:new\s+)?user`), 0.8},
			{regexp.MustCompile(`(?i)set\s+up\s+user`), 0.7},
			{regexp.MustCompile(`(?i)create\s+account`), 0.7},
		},
		EntityExtractors: []Pattern{
			{regexp.MustCompile(`(?i)user\s+(?:with\s+name\s+)?(\w+)`), 1.0},
			{regexp.MustCompile(`(?i)for\s+(\w+)`), 0.8},
			{regexp.MustCompile(`(?i)with\s+name\s+(\w+)`), 1.0},
		},
		EntityFieldName: "name",
	}

	// Delete user patterns
	cp.intentPatterns["delete_user"] = IntentPattern{
		Patterns: []Pattern{
			{regexp.MustCompile(`(?i)delete\s+user`), 1.0},
			{regexp.MustCompile(`(?i)remove\s+user`), 0.9},
			{regexp.MustCompile(`(?i)deactivate\s+user`), 0.8},
			{regexp.MustCompile(`(?i)erase\s+user`), 0.8},
			{regexp.MustCompile(`(?i)delete\s+account`), 0.7},
		},
		EntityExtractors: []Pattern{
			{regexp.MustCompile(`(?i)user\s+(\w+)`), 1.0},
			{regexp.MustCompile(`(?i)for\s+(\w+)`), 0.8},
		},
		EntityFieldName: "name",
	}

	// Activate patterns
	cp.intentPatterns["activate"] = IntentPattern{
		Patterns: []Pattern{
			{regexp.MustCompile(`(?i)activate`), 1.0},
			{regexp.MustCompile(`(?i)enable`), 0.9},
			{regexp.MustCompile(`(?i)turn\s+on`), 0.8},
		},
		EntityExtractors: []Pattern{
			{regexp.MustCompile(`(?i)activate\s+(?:feature\s+)?(\w+(?:\s+\w+)*)`), 1.0},
			{regexp.MustCompile(`(?i)enable\s+(\w+(?:\s+\w+)*)`), 1.0},
			{regexp.MustCompile(`(?i)turn\s+on\s+(\w+(?:\s+\w+)*)`), 1.0},
		},
		EntityFieldName: "feature",
	}

	// Deactivate patterns
	cp.intentPatterns["deactivate"] = IntentPattern{
		Patterns: []Pattern{
			{regexp.MustCompile(`(?i)deactivate`), 1.0},
			{regexp.MustCompile(`(?i)disable`), 0.9},
			{regexp.MustCompile(`(?i)turn\s+off`), 0.8},
		},
		EntityExtractors: []Pattern{
			{regexp.MustCompile(`(?i)deactivate\s+(?:feature\s+)?(\w+(?:\s+\w+)*)`), 1.0},
			{regexp.MustCompile(`(?i)disable\s+(\w+(?:\s+\w+)*)`), 1.0},
			{regexp.MustCompile(`(?i)turn\s+off\s+(\w+(?:\s+\w+)*)`), 1.0},
		},
		EntityFieldName: "feature",
	}

	return cp
}

// calculateIntentScore computes a confidence score for a specific intent
func (cp *CommandParser) calculateIntentScore(command string, intentPatterns IntentPattern) float64 {
	maxScore := 0.0

	// Check each pattern and keep the highest score
	for _, patternInfo := range intentPatterns.Patterns {
		if patternInfo.Pattern.MatchString(command) {
			if patternInfo.Weight > maxScore {
				maxScore = patternInfo.Weight
			}
		}
	}

	// Enhance score with word features (simple bag-of-words approach)
	words := strings.Fields(strings.ToLower(command))
	wordScore := 0.0
	for _, word := range words {
		if _, exists := cp.vocabulary[word]; exists {
			wordScore += 0.05 // Small boost for each relevant vocabulary word
		}
	}

	// Combine scores, capping at 1.0
	totalScore := math.Min(maxScore+wordScore, 1.0)
	return totalScore
}

// extractEntityData extracts entity information from the command
func (cp *CommandParser) extractEntityData(command string, intentPattern IntentPattern) (map[string]interface{}, float64) {
	data := make(map[string]interface{})
	maxConfidence := 0.0

	// Try each entity extractor pattern
	for _, extractorInfo := range intentPattern.EntityExtractors {
		matches := extractorInfo.Pattern.FindStringSubmatch(command)
		if len(matches) > 1 {
			entity := strings.TrimSpace(matches[1])
			data[intentPattern.EntityFieldName] = entity
			maxConfidence = math.Max(maxConfidence, extractorInfo.Weight)
			break
		}
	}

	// Apply any additional parsers
	for _, parser := range intentPattern.AdditionalParsers {
		parser(command, data)
	}

	return data, maxConfidence
}

// ParseCommand parses a command string into structured data
func (cp *CommandParser) ParseCommand(commandText string) CommandResult {
	// Clean up the command string
	command := strings.TrimSpace(commandText)

	bestIntent := "unknown"
	bestScore := 0.0
	var bestData map[string]interface{}

	// Find the intent with the highest score
	for intent, intentPattern := range cp.intentPatterns {
		score := cp.calculateIntentScore(command, intentPattern)
		if score > bestScore {
			bestScore = score
			bestIntent = intent
		}
	}

	// If we found a valid intent
	if bestIntent != "unknown" && bestScore > 0 {
		// Extract entities
		data, entityConfidence := cp.extractEntityData(command, cp.intentPatterns[bestIntent])
		bestData = data

		// Adjust confidence based on entity extraction
		if entityConfidence > 0 {
			// Average the intent confidence with entity confidence
			bestScore = (bestScore + entityConfidence) / 2
		} else {
			// Reduce confidence if we couldn't extract expected entities
			bestScore *= 0.7
		}

		return CommandResult{
			What:       bestIntent,
			Data:       bestData,
			Confidence: bestScore,
		}
	}

	// If no intent was found
	return CommandResult{
		What:       "unknown",
		Data:       make(map[string]interface{}),
		Confidence: 0.0,
		Error:      "Command not recognized",
	}
}

func main() {
	// Define command-line flags
	interactive := flag.Bool("interactive", false, "Run in interactive mode")
	flag.BoolVar(interactive, "i", false, "Run in interactive mode (shorthand)")
	flag.Parse()

	cp := NewCommandParser()

	if *interactive {
		fmt.Println("=== Advanced Command Parser ===")
		fmt.Println("Type 'exit' or 'quit' to end the session")
		fmt.Println("================================\n")

		scanner := bufio.NewScanner(os.Stdin)
		for {
			fmt.Print("\nEnter a command: ")
			if !scanner.Scan() {
				break
			}

			userInput := scanner.Text()
			if userInput == "exit" || userInput == "quit" {
				fmt.Println("Goodbye!")
				break
			}

			result := cp.ParseCommand(userInput)
			jsonResult, err := json.MarshalIndent(result, "", "  ")
			if err != nil {
				fmt.Printf("Error encoding JSON: %v\n", err)
				continue
			}

			fmt.Println(string(jsonResult))
		}

		if err := scanner.Err(); err != nil {
			fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
		}
	} else {
		// Process a single command
		fmt.Print("Enter a command: ")
		scanner := bufio.NewScanner(os.Stdin)
		if scanner.Scan() {
			userInput := scanner.Text()
			result := cp.ParseCommand(userInput)
			jsonResult, err := json.MarshalIndent(result, "", "  ")
			if err != nil {
				fmt.Printf("Error encoding JSON: %v\n", err)
				return
			}

			fmt.Println(string(jsonResult))
		}

		if err := scanner.Err(); err != nil {
			fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
		}
	}
}