import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

// Komponen Ikon & Spinner
const UploadIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
    <polyline points="17 8 12 3 7 8"></polyline>
    <line x1="12" y1="3" x2="12" y2="15"></line>
  </svg>
);

const ChatIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" >
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
    </svg>
);

const Spinner = () => <div className="spinner"></div>;

function App() {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [lang, setLang] = useState('id');
  const [summary, setSummary] = useState(null);
  const [documentId, setDocumentId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  
  const [question, setQuestion] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isAnswering, setIsAnswering] = useState(false);
  
  const fileInputRef = useRef(null);

  const handleFileChange = (selectedFile) => {
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setError('');
    } else {
      setError('Harap pilih file dengan format .pdf');
    }
  };

  const resetSummarizer = () => {
    setFile(null);
    setFileName('');
    setSummary(null);
    setDocumentId(null);
    setError('');
  };

  const handleSummarize = async () => {
    if (!file) {
      setError('Silakan pilih file PDF terlebih dahulu.');
      return;
    }
    setIsLoading(true);
    setError('');
    setSummary(null);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('lang', lang);
    try {
      const response = await axios.post('http://localhost:8000/summarize', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setSummary(response.data.structured_summary);
      setDocumentId(response.data.document_id);
    } catch (err) {
      const errorMessage = err.response?.data?.detail || 'Terjadi kesalahan tidak diketahui.';
      setError(`Gagal meringkas: ${errorMessage}`);
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleAskQuestion = async (e) => {
    e.preventDefault();
    if (!question.trim() || isAnswering) return;
    const newChatHistory = [...chatHistory, { role: 'user', content: question }];
    setChatHistory(newChatHistory);
    const currentQuestion = question;
    setQuestion('');
    setIsAnswering(true);
    try {
      const response = await axios.post('http://localhost:8000/qa', {
        document_id: documentId,
        question: currentQuestion,
        lang: lang
      });
      setChatHistory([...newChatHistory, { role: 'bot', content: response.data.answer }]);
    } catch (err) {
      const errorMessage = err.response?.data?.detail || 'Tidak bisa mendapatkan jawaban.';
      setChatHistory([...newChatHistory, { role: 'bot', content: `Error: ${errorMessage}` }]);
    } finally {
      setIsAnswering(false);
    }
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>AI Research Assistant</h1>
        <p>Ringkas dan tanyakan apa pun dari paper penelitian Anda.</p>
      </header>
      
      <main className="main-content">
        {/* Kolom Kiri */}
        <div className="card-container">
          <div className="card">
            <h2 className="card-title">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
              Summarizer
            </h2>
            
            {/* --- PERBAIKAN: MENAMBAHKAN WRAPPER --- */}
            <div className="card-content-wrapper">
              {!summary ? (
                <>
                  <div 
                    className="file-drop-area"
                    onClick={() => fileInputRef.current.click()}
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={(e) => {
                      e.preventDefault();
                      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                        handleFileChange(e.dataTransfer.files[0]);
                      }
                    }}
                  >
                    <input type="file" accept=".pdf" ref={fileInputRef} onChange={(e) => handleFileChange(e.target.files[0])} style={{ display: 'none' }} />
                    <UploadIcon />
                    <p>{fileName || 'Klik atau jatuhkan file PDF di sini'}</p>
                  </div>

                  <div className="language-selector">
                    <span>Bahasa Ringkasan:</span>
                    <div className="language-toggle">
                      <button className={lang === 'id' ? 'active' : ''} onClick={() => setLang('id')}>
                        ID
                      </button>
                      <button className={lang === 'en' ? 'active' : ''} onClick={() => setLang('en')}>
                        EN
                      </button>
                    </div>
                  </div>

                  <button onClick={handleSummarize} disabled={isLoading || !file}>
                    {isLoading ? <Spinner /> : 'Buat Ringkasan'}
                  </button>
                  {error && <p className="error-message">{error}</p>}
                </>
              ) : (
                <div className="summary-results">
                  <h3>Hasil Ringkasan untuk: <strong>{fileName}</strong></h3>
                  {Object.entries(summary).map(([key, value]) => (
                    <div key={key} className="summary-item">
                      <h4>{key}</h4>
                      <p>{value || 'N/A'}</p>
                    </div>
                  ))}
                  <button onClick={resetSummarizer} className="secondary-button">
                    Ringkas Dokumen Lain
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Kolom Kanan */}
        <div className="card-container">
          <div className="card card-qa">
             <h2 className="card-title">
              Q&A Chat
            </h2>
            {/* --- PERBAIKAN: MENAMBAHKAN WRAPPER DAN MEMINDAHKAN KONTEN KE DALAMNYA --- */}
            <div className="card-content-wrapper">
              {/*<p className="qa-context">
                  {documentId ? `Bertanya tentang: ${fileName}` : 'Bertanya tentang pengetahuan umum'}
              </p>*/}
              <div className="chat-window">
                {chatHistory.length === 0 && (
                  <div className="empty-chat-placeholder">
                    <ChatIcon />
                    <p>Tanyakan apa saja.</p>
                    <span>Jika Anda mengunggah dokumen, pertanyaan akan difokuskan pada konten tersebut.</span>
                  </div>
                )}
                {chatHistory.map((msg, index) => (
                  <div key={index} className={`chat-message ${msg.role}`}>
                    <p>{msg.content}</p>
                  </div>
                ))}
                {isAnswering && <div className="chat-message bot"><Spinner/></div>}
              </div>
            </div>
            <form onSubmit={handleAskQuestion} className="chat-input">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ketik pertanyaan Anda..."
                disabled={isAnswering}
              />
              <button type="submit" disabled={isAnswering || !question.trim()}>Kirim</button>
            </form>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;

