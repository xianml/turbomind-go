package turbomind

import (
	"fmt"
	"path/filepath"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

// Tokenizer wraps the sugarme/tokenizer library for TurboMind
type Tokenizer struct {
	tokenizer *tokenizer.Tokenizer
	bosToken  int
	eosToken  int
	padToken  int
}

// TokenizerConfig holds tokenizer configuration
type TokenizerConfig struct {
	TokenizerPath string // Path to tokenizer.json file
	BosToken      int    // Beginning of sequence token
	EosToken      int    // End of sequence token  
	PadToken      int    // Padding token
}

// NewTokenizer creates a new tokenizer from the model directory
func NewTokenizer(modelDir string) (*Tokenizer, error) {
	tokenizerPath := filepath.Join(modelDir, "tokenizer.json")
	
	tk, err := pretrained.FromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer from %s: %v", tokenizerPath, err)
	}
	
	return &Tokenizer{
		tokenizer: tk,
		bosToken:  1,     // <s>
		eosToken:  32000, // <|endoftext|>
		padToken:  32000, // <|endoftext|> (same as EOS for this model)
	}, nil
}

// NewTokenizerWithConfig creates a tokenizer with custom configuration
func NewTokenizerWithConfig(config *TokenizerConfig) (*Tokenizer, error) {
	tk, err := pretrained.FromFile(config.TokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer from %s: %v", config.TokenizerPath, err)
	}
	
	return &Tokenizer{
		tokenizer: tk,
		bosToken:  config.BosToken,
		eosToken:  config.EosToken,
		padToken:  config.PadToken,
	}, nil
}


// Encode encodes text to token IDs
func (t *Tokenizer) Encode(text string, addSpecialTokens bool) ([]int, error) {
	encoding, err := t.tokenizer.EncodeSingle(text, addSpecialTokens)
	if err != nil {
		return nil, fmt.Errorf("failed to encode text: %v", err)
	}
	
	// Get token IDs
	tokens := encoding.Ids
	result := make([]int, len(tokens))
	for i, token := range tokens {
		result[i] = int(token)
	}
	
	return result, nil
}

// EncodeWithBOS encodes text and adds BOS token at the beginning
func (t *Tokenizer) EncodeWithBOS(text string) ([]int, error) {
	tokens, err := t.Encode(text, false)
	if err != nil {
		return nil, err
	}
	
	// Add BOS token at the beginning
	result := make([]int, 0, len(tokens)+1)
	result = append(result, t.bosToken)
	result = append(result, tokens...)
	
	return result, nil
}

// Decode decodes token IDs back to text
func (t *Tokenizer) Decode(tokens []int, skipSpecialTokens bool) (string, error) {
	text := t.tokenizer.Decode(tokens, skipSpecialTokens)
	return text, nil
}

// GetVocabSize returns the vocabulary size
func (t *Tokenizer) GetVocabSize() int {
	return int(t.tokenizer.GetVocabSize(true))
}

// GetBOSToken returns the BOS token ID
func (t *Tokenizer) GetBOSToken() int {
	return t.bosToken
}

// GetEOSToken returns the EOS token ID
func (t *Tokenizer) GetEOSToken() int {
	return t.eosToken
}

// GetPadToken returns the pad token ID
func (t *Tokenizer) GetPadToken() int {
	return t.padToken
}

// ConvertTokensToString converts tokens to string (useful for streaming)
func (t *Tokenizer) ConvertTokensToString(tokens []int) (string, error) {
	return t.Decode(tokens, true)
}

// TokenizeAndPad tokenizes text and pads to specified length
func (t *Tokenizer) TokenizeAndPad(text string, maxLength int, addSpecialTokens bool) ([]int, error) {
	tokens, err := t.Encode(text, addSpecialTokens)
	if err != nil {
		return nil, err
	}
	
	// Truncate if too long
	if len(tokens) > maxLength {
		tokens = tokens[:maxLength]
	}
	
	// Pad if too short
	for len(tokens) < maxLength {
		tokens = append(tokens, t.padToken)
	}
	
	return tokens, nil
}

// Close releases tokenizer resources
func (t *Tokenizer) Close() {
	// sugarme/tokenizer doesn't require explicit cleanup
	t.tokenizer = nil
}

// GetTokenText returns the text representation of a token ID
func (t *Tokenizer) GetTokenText(tokenID int) (string, error) {
	text, ok := t.tokenizer.IdToToken(tokenID)
	if !ok {
		return "", fmt.Errorf("failed to get token text for ID %d", tokenID)
	}
	return text, nil
}

// GetTokenID returns the token ID for a text token
func (t *Tokenizer) GetTokenID(token string) (int, error) {
	id, ok := t.tokenizer.TokenToId(token)
	if !ok {
		return -1, fmt.Errorf("token '%s' not found in vocabulary", token)
	}
	return id, nil
}

// EncodeWithOffset encodes text and returns tokens with their text offsets
func (t *Tokenizer) EncodeWithOffset(text string, addSpecialTokens bool) ([]int, [][2]int, error) {
	encoding, err := t.tokenizer.EncodeSingle(text, addSpecialTokens)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to encode text with offsets: %v", err)
	}
	
	tokens := encoding.Ids
	offsets := encoding.Offsets
	
	// Convert uint32 to int for tokens
	result := make([]int, len(tokens))
	for i, token := range tokens {
		result[i] = int(token)
	}
	
	// Convert offsets to [][2]int format
	resultOffsets := make([][2]int, len(offsets))
	for i, offset := range offsets {
		resultOffsets[i] = [2]int{offset[0], offset[1]}
	}
	
	return result, resultOffsets, nil
}