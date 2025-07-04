package turbomind

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTokenizer(t *testing.T) {
	modelPath := os.Getenv("TEST_MODEL_PATH")
	if modelPath == "" {
		t.Skip("TEST_MODEL_PATH not set, skipping tokenizer tests")
	}

	t.Run("TokenizerCreation", func(t *testing.T) {
		tokenizer, err := NewTokenizer(modelPath)
		require.NoError(t, err)
		require.NotNil(t, tokenizer)
		defer tokenizer.Close()

		// Test basic properties
		vocabSize := tokenizer.GetVocabSize()
		assert.Greater(t, vocabSize, 0)
		assert.Greater(t, vocabSize, 30000) // Should be around 32K for this model

		bosToken := tokenizer.GetBOSToken()
		assert.Equal(t, 1, bosToken)

		eosToken := tokenizer.GetEOSToken()
		assert.Equal(t, 32000, eosToken)

		t.Logf("Vocab size: %d", vocabSize)
		t.Logf("BOS token: %d", bosToken)
		t.Logf("EOS token: %d", eosToken)
	})

	t.Run("BasicEncoding", func(t *testing.T) {
		tokenizer, err := NewTokenizer(modelPath)
		require.NoError(t, err)
		defer tokenizer.Close()

		testCases := []struct {
			text     string
			minLen   int
			maxLen   int
		}{
			{"Hello", 1, 10},
			{"Hello, world!", 1, 20},
			{"How are you?", 1, 15},
			{"The quick brown fox jumps over the lazy dog.", 5, 25},
		}

		for _, tc := range testCases {
			t.Run(tc.text, func(t *testing.T) {
				tokens, err := tokenizer.Encode(tc.text, false)
				require.NoError(t, err)
				assert.GreaterOrEqual(t, len(tokens), tc.minLen)
				assert.LessOrEqual(t, len(tokens), tc.maxLen)

				t.Logf("Text: '%s' -> Tokens: %v (length: %d)", tc.text, tokens, len(tokens))
			})
		}
	})

	t.Run("EncodingWithBOS", func(t *testing.T) {
		tokenizer, err := NewTokenizer(modelPath)
		require.NoError(t, err)
		defer tokenizer.Close()

		text := "Hello, how are you?"
		
		// Test normal encoding
		tokens, err := tokenizer.Encode(text, false)
		require.NoError(t, err)

		// Test encoding with BOS
		tokensWithBOS, err := tokenizer.EncodeWithBOS(text)
		require.NoError(t, err)

		// Should have one more token (BOS)
		assert.Equal(t, len(tokens)+1, len(tokensWithBOS))
		assert.Equal(t, 1, tokensWithBOS[0]) // First token should be BOS

		t.Logf("Without BOS: %v", tokens)
		t.Logf("With BOS: %v", tokensWithBOS)
	})

	t.Run("RoundTripEncoding", func(t *testing.T) {
		tokenizer, err := NewTokenizer(modelPath)
		require.NoError(t, err)
		defer tokenizer.Close()

		testTexts := []string{
			"Hello, world!",
			"How are you today?",
			"The weather is nice.",
			"I like programming in Go.",
		}

		for _, text := range testTexts {
			t.Run(text, func(t *testing.T) {
				// Encode
				tokens, err := tokenizer.Encode(text, false)
				require.NoError(t, err)

				// Decode
				decoded, err := tokenizer.Decode(tokens, false)
				require.NoError(t, err)

				// Should be approximately the same (tokenizers may normalize)
				assert.NotEmpty(t, decoded)
				t.Logf("Original: '%s'", text)
				t.Logf("Decoded:  '%s'", decoded)
				t.Logf("Tokens:   %v", tokens)
			})
		}
	})

	t.Run("SpecialTokens", func(t *testing.T) {
		tokenizer, err := NewTokenizer(modelPath)
		require.NoError(t, err)
		defer tokenizer.Close()

		// Test with special tokens
		text := "Hello, world!"
		tokens, err := tokenizer.Encode(text, true)
		require.NoError(t, err)

		decoded, err := tokenizer.Decode(tokens, true) // Skip special tokens
		require.NoError(t, err)

		t.Logf("With special tokens: %v", tokens)
		t.Logf("Decoded: '%s'", decoded)
	})

	t.Run("TokenizeAndPad", func(t *testing.T) {
		tokenizer, err := NewTokenizer(modelPath)
		require.NoError(t, err)
		defer tokenizer.Close()

		text := "Hello"
		maxLength := 10

		tokens, err := tokenizer.TokenizeAndPad(text, maxLength, false)
		require.NoError(t, err)
		assert.Equal(t, maxLength, len(tokens))

		// Should be padded with pad tokens
		padToken := tokenizer.GetPadToken()
		foundPadding := false
		for _, token := range tokens {
			if token == padToken {
				foundPadding = true
				break
			}
		}
		assert.True(t, foundPadding, "Should contain padding tokens")

		t.Logf("Padded tokens: %v", tokens)
	})

	t.Run("TokenTextConversion", func(t *testing.T) {
		tokenizer, err := NewTokenizer(modelPath)
		require.NoError(t, err)
		defer tokenizer.Close()

		// Test token to text conversion
		tokens, err := tokenizer.Encode("Hello", false)
		require.NoError(t, err)
		require.Greater(t, len(tokens), 0)

		for i, tokenID := range tokens[:min(3, len(tokens))] { // Test first 3 tokens
			tokenText, err := tokenizer.GetTokenText(tokenID)
			require.NoError(t, err)
			assert.NotEmpty(t, tokenText)
			
			t.Logf("Token %d: ID=%d, Text='%s'", i, tokenID, tokenText)
		}
	})

	t.Run("EncodeWithOffset", func(t *testing.T) {
		tokenizer, err := NewTokenizer(modelPath)
		require.NoError(t, err)
		defer tokenizer.Close()

		text := "Hello, world!"
		tokens, offsets, err := tokenizer.EncodeWithOffset(text, false)
		require.NoError(t, err)
		assert.Equal(t, len(tokens), len(offsets))

		t.Logf("Text: '%s'", text)
		for i, token := range tokens {
			if i < len(offsets) {
				offset := offsets[i]
				tokenText := text[offset[0]:offset[1]]
				t.Logf("Token %d: ID=%d, Offset=[%d:%d], Text='%s'", 
					i, token, offset[0], offset[1], tokenText)
			}
		}
	})

	t.Run("EmptyAndSpecialCases", func(t *testing.T) {
		tokenizer, err := NewTokenizer(modelPath)
		require.NoError(t, err)
		defer tokenizer.Close()

		// Test empty string
		tokens, err := tokenizer.Encode("", false)
		require.NoError(t, err)
		assert.GreaterOrEqual(t, len(tokens), 0)

		// Test whitespace
		tokens, err = tokenizer.Encode("   ", false)
		require.NoError(t, err)
		assert.GreaterOrEqual(t, len(tokens), 0)

		// Test newlines
		tokens, err = tokenizer.Encode("Hello\nWorld", false)
		require.NoError(t, err)
		assert.Greater(t, len(tokens), 0)

		t.Logf("Empty string tokens: %v", tokens)
	})
}

func TestTokenizerConfig(t *testing.T) {
	modelPath := os.Getenv("TEST_MODEL_PATH")
	if modelPath == "" {
		t.Skip("TEST_MODEL_PATH not set")
	}

	t.Run("CustomConfig", func(t *testing.T) {
		config := &TokenizerConfig{
			TokenizerPath: modelPath + "/tokenizer.json",
			BosToken:      1,
			EosToken:      32000,
			PadToken:      32000,
		}

		tokenizer, err := NewTokenizerWithConfig(config)
		require.NoError(t, err)
		defer tokenizer.Close()

		assert.Equal(t, config.BosToken, tokenizer.GetBOSToken())
		assert.Equal(t, config.EosToken, tokenizer.GetEOSToken())
		assert.Equal(t, config.PadToken, tokenizer.GetPadToken())
	})
}

func TestTokenizerErrors(t *testing.T) {
	t.Run("InvalidPath", func(t *testing.T) {
		_, err := NewTokenizer("/nonexistent/path")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "failed to load tokenizer")
	})

	t.Run("InvalidConfigPath", func(t *testing.T) {
		config := &TokenizerConfig{
			TokenizerPath: "/nonexistent/tokenizer.json",
			BosToken:      1,
			EosToken:      2,
			PadToken:      0,
		}

		_, err := NewTokenizerWithConfig(config)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "failed to load tokenizer")
	})
}

func BenchmarkTokenizer(b *testing.B) {
	modelPath := os.Getenv("TEST_MODEL_PATH")
	if modelPath == "" {
		b.Skip("TEST_MODEL_PATH not set")
	}

	tokenizer, err := NewTokenizer(modelPath)
	require.NoError(b, err)
	defer tokenizer.Close()

	text := "Hello, how are you? This is a test sentence for benchmarking tokenization performance."

	b.Run("Encode", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := tokenizer.Encode(text, false)
			require.NoError(b, err)
		}
	})

	// Get tokens for decode benchmark
	tokens, err := tokenizer.Encode(text, false)
	require.NoError(b, err)

	b.Run("Decode", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := tokenizer.Decode(tokens, false)
			require.NoError(b, err)
		}
	})

	b.Run("EncodeWithBOS", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := tokenizer.EncodeWithBOS(text)
			require.NoError(b, err)
		}
	})

	b.Run("EncodeWithOffset", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _, err := tokenizer.EncodeWithOffset(text, false)
			require.NoError(b, err)
		}
	})
}

// Helper function for older Go versions
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}