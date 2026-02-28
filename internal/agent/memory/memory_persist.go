package memory

import (
	"database/sql"
	"database/sql/driver"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sync"

	"modernc.org/sqlite"

	"github.com/local/picobot/internal/agent/memory/onnx"
	"github.com/local/picobot/internal/config"
	"github.com/local/picobot/internal/session"
)

const isNormalizedPrecisionTolerance = 1e-6

type EmbedProvider interface {
	Embed(items string) ([]float32, error)
}

type MemoryPersist struct {
	embedder  EmbedProvider
	mu        sync.Mutex
	db        *sql.DB
	threshold float32
	topk      int
}

func NewPersistMemory(memConf *config.MemoryConfig) *MemoryPersist {
	if memConf.EmbedType != "onnx" {
		logf("Unknown embed type: %s", memConf.EmbedType)
		return nil
	}

	home, _ := os.UserHomeDir()
	memConf.DbPath = expandPath(memConf.DbPath, home)
	memConf.ONNXModelPath = expandPath(memConf.ONNXModelPath, home)
	memConf.ONNXTokenizerPath = expandPath(memConf.ONNXTokenizerPath, home)

	var mem MemoryPersist
	onnxemb, err := NewONNXEmbedder(&onnx.ModelConfig{
		Path:                memConf.ONNXModelPath,
		TokenizerPath:       memConf.ONNXTokenizerPath,
		NormalizeEmbeddings: true,
		BatchSize:           32,
	})
	if err != nil {
		logf("Failed to initialize ONNX embedder: %v", err)
		return nil
	}

	db, err := initDB(memConf.DbPath)
	if err != nil {
		logf("Failed to initialize memory database: %v", err)
		return nil
	}

	err = initSchema(db)
	if err != nil {
		logf("Failed to initialize memory schema: %v", err)
		return nil
	}

	if memConf.Threshold <= 0 {
		memConf.Threshold = 0.87
	}
	if memConf.TopK <= 0 {
		memConf.TopK = 10
	}
	mem.db = db
	mem.embedder = onnxemb
	mem.threshold = memConf.Threshold
	mem.topk = memConf.TopK
	return &mem
}

// StoreHistory saves a memory item to the database with its embedding.
func (m *MemoryPersist) StoreHistory(channelID, role, content, timestamp string) error {
	embedding, err := m.embedder.Embed(content)
	if err != nil {
		return fmt.Errorf("embedding content: %w", err)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	tx, err := m.db.Begin()
	if err != nil {
		return fmt.Errorf("SQL begin transaction: %w", err)
	}

	stmt, err := tx.Prepare(`
        INSERT INTO history (channel_id, role, content, timestamp, embedding)
        VALUES (?, ?, ?, ?, ?)
    `)
	if err != nil {
		tx.Rollback()
		return fmt.Errorf("prepare statement: %w", err)
	}
	defer stmt.Close()

	if _, err := stmt.Exec(channelID, role, content, timestamp, floatsToBytes(embedding)); err != nil {
		tx.Rollback()
		return fmt.Errorf("exec insert: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit transaction: %w", err)
	}
	return nil
}

// Store in batch for efficiency
func (m *MemoryPersist) BatchStoreHistory(channelID string, items []*session.Message) error {
	if len(items) == 0 {
		return nil
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	tx, err := m.db.Begin()
	if err != nil {
		return err
	}
	stmt, err := tx.Prepare(`
        INSERT INTO history (channel_id, role, content, timestamp, embedding)
        VALUES (?, ?, ?, ?, ?)
    `)
	if err != nil {
		tx.Rollback()
		return err
	}
	defer stmt.Close()

	for _, it := range items {
		emb, err := m.embedder.Embed(it.Content)
		if err != nil {
			tx.Rollback()
			return err
		}
		if _, err := stmt.Exec(channelID, it.Role, it.Content, it.Timestamp, floatsToBytes(emb)); err != nil {
			tx.Rollback()
			return err
		}
	}
	return tx.Commit()
}

// QueryHistory retrieves the most relevant memory items for a given query.
func (m *MemoryPersist) QueryHistory(channelID, query string, topk int) ([]MemoryItem, error) {

	embedding, err := m.embedder.Embed(query)
	if err != nil {
		return nil, fmt.Errorf("embedding query: %w", err)
	}

	if topk <= 0 {
		topk = m.topk
	}
	sqlStr := `
SELECT role, content, timestamp, cosine_similarity(embedding, ?) AS similarity
FROM history WHERE similarity >= ?
`
	args := []interface{}{floatsToBytes(embedding), m.threshold}

	if channelID != "" {
		sqlStr += "AND channel_id = ?\n"
		args = append(args, channelID)
	}
	sqlStr += "ORDER BY similarity DESC\nLIMIT ?"
	args = append(args, topk)

	rows, err := m.db.Query(sqlStr, args...)
	if err != nil {
		return nil, fmt.Errorf("querying memory: %w", err)
	}
	defer rows.Close()

	var results []MemoryItem
	for rows.Next() {
		var item MemoryItem
		if err := rows.Scan(&item.Role, &item.Text, &item.Timestamp, &item.Similarity); err != nil {
			return nil, fmt.Errorf("scanning row: %w", err)
		}
		results = append(results, item)
	}
	return results, nil
}

func (m *MemoryPersist) Embed(items string) ([]float32, error) {
	if m.embedder == nil {
		return nil, fmt.Errorf("embedder is not initialized")
	}
	return m.embedder.Embed(items)
}

func (m *MemoryPersist) Close() error {
	if m.db != nil {
		return m.db.Close()
	}
	return nil
}

func logf(format string, args ...interface{}) {
	log.Printf("[Memory] "+format, args...)
}

func initDB(dbPath string) (*sql.DB, error) {
	err := sqlite.RegisterFunction("cosine_similarity", &sqlite.FunctionImpl{
		NArgs:         2,
		Deterministic: true,
		Scalar:        cosineSimilarityScalar,
	})
	if err != nil {
		return nil, fmt.Errorf("registering function: %w", err)
	}

	err = sqlite.RegisterFunction("cosine_distance", &sqlite.FunctionImpl{
		NArgs:         2,
		Deterministic: true,
		Scalar:        cosineDistanceScalar,
	})
	if err != nil {
		return nil, fmt.Errorf("registering function: %w", err)
	}

	db_path := fmt.Sprintf("file:%s", dbPath)
	db, err := sql.Open("sqlite", db_path)
	if err != nil {
		return nil, fmt.Errorf("opening sqlite db: %w", err)
	}

	return db, err
}

func initSchema(db *sql.DB) error {
	schema := `
CREATE TABLE IF NOT EXISTS history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id   TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    embedding   BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_channel_id ON history(channel_id);
`
	_, err := db.Exec(schema)
	return err
}

func cosineSimilarityScalar(_ *sqlite.FunctionContext, args []driver.Value) (driver.Value, error) {
	if len(args) != 2 {
		return nil, fmt.Errorf("cosine_similarity expects exactly 2 BLOB arguments")
	}
	//log.Printf("Received blobs of lengths: %d and %d\n", len(args[0].([]byte)), len(args[1].([]byte)))
	qBlob, ok1 := args[0].([]byte)
	rowBlob, ok2 := args[1].([]byte)
	if !ok1 || !ok2 {
		return nil, fmt.Errorf("both arguments must be BLOB")
	}

	if len(qBlob)%4 != 0 || len(rowBlob)%4 != 0 {
		return nil, fmt.Errorf("blobs must contain whole float32 values")
	}
	if len(qBlob) != len(rowBlob) {
		paddedLength := max(len(qBlob), len(rowBlob))
		//log.Printf("Padding vectors to length %d bytes\n", paddedLength)
		qBlob = append(qBlob, make([]byte, paddedLength-len(qBlob))...)
		rowBlob = append(rowBlob, make([]byte, paddedLength-len(rowBlob))...)
		// return nil, fmt.Errorf("vector dimension mismatch")
	}

	if len(qBlob) == 0 || len(rowBlob) == 0 {
		return 0.0, nil // treat empty as maximally dissimilar
	}

	if len(qBlob) != len(rowBlob) {
		return nil, fmt.Errorf("vector dimension mismatch")
	}

	qVec := bytesToFloats(qBlob)
	rowVec := bytesToFloats(rowBlob)

	var dot float64
	for i := range qVec {
		dot += float64(qVec[i] * rowVec[i])
	}

	return dot, nil
}

func cosineDistanceScalar(_ *sqlite.FunctionContext, args []driver.Value) (driver.Value, error) {
	if len(args) != 2 {
		return nil, fmt.Errorf("cosine_distance expects exactly 2 BLOB arguments")
	}

	qBlob, ok1 := args[0].([]byte)
	rowBlob, ok2 := args[1].([]byte)
	if !ok1 || !ok2 {
		return nil, fmt.Errorf("both arguments must be BLOB")
	}

	if len(qBlob) == 0 || len(rowBlob) == 0 {
		return 1.0, nil // treat empty as maximally distant
	}

	if len(qBlob)%4 != 0 || len(rowBlob)%4 != 0 {
		return nil, fmt.Errorf("BLOB length must be multiple of 4 (float32)")
	}

	if len(qBlob) != len(rowBlob) {
		paddedLength := max(len(qBlob), len(rowBlob))
		log.Printf("Padding vectors to length %d bytes\n", paddedLength)
		qBlob = append(qBlob, make([]byte, paddedLength-len(qBlob))...)
		rowBlob = append(rowBlob, make([]byte, paddedLength-len(rowBlob))...)
		// return nil, fmt.Errorf("vector dimension mismatch")
	}

	if len(qBlob) == 0 || len(rowBlob) == 0 {
		return 1.0, nil // treat empty as maximally distant
	}

	if len(qBlob) != len(rowBlob) {
		return nil, fmt.Errorf("vector dimension mismatch")
	}

	// Convert blobs → []float32
	qVec := bytesToFloats(qBlob)
	rowVec := bytesToFloats(rowBlob)

	var dot float64
	for i := range qVec {
		dot += float64(qVec[i] * rowVec[i])
	}

	cosSim := dot // Assuming vectors are normalized, cosine similarity is just the dot product
	cosDist := 1.0 - cosSim

	return cosDist, nil
}

func bytesToFloats(b []byte) []float32 {
	if len(b)%4 != 0 {
		panic("invalid blob length")
	}
	vec := make([]float32, len(b)/4)
	for i := range vec {
		vec[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return vec
}

func floatsToBytes(vec []float32) []byte {
	buf := make([]byte, len(vec)*4)
	for i, f := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

func isNormalized(v []float32) bool {
	var sqSum float64
	for _, val := range v {
		sqSum += float64(val) * float64(val)
	}
	magnitude := math.Sqrt(sqSum)
	return math.Abs(magnitude-1) < isNormalizedPrecisionTolerance
}

func normalizeVector(v []float32) []float32 {
	var norm float32
	for _, val := range v {
		norm += val * val
	}
	norm = float32(math.Sqrt(float64(norm)))

	res := make([]float32, len(v))
	for i, val := range v {
		res[i] = val / norm
	}

	return res
}

func expandPath(path, home string) string {
	if path == "" {
		return path
	}
	if path[0] == '~' {
		return filepath.Join(home, path[1:])
	}
	return path
}
