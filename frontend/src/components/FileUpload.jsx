import React, { useState, useRef } from 'react';

const FileUpload = ({ onUploadSuccess }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState('');
  const [isProcessing, setIsProcessing] = useState(false); 
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const validateFile = (file) => {
    if (file && file.type === 'application/pdf') {
      setSelectedFile(file);
      setError('');
      return true;
    }
    setError('Invalid file type. Please upload a PDF.');
    return false;
  };

  const handleFileChange = (e) => validateFile(e.target.files[0]);
  const handleDragOver = (e) => { e.preventDefault(); setIsDragOver(true); };
  const handleDragLeave = (e) => { e.preventDefault(); setIsDragOver(false); };
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    validateFile(e.dataTransfer.files[0]);
  };
  const handleDropZoneClick = () => fileInputRef.current.click();

  const API_URL = import.meta.env.VITE_API_BASE_URL || "https://127.0.0.1:8000"
  
  const handleUpload = async () => {
    if (!selectedFile) return;
    setIsProcessing(true);
    setError('');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_URL}/ingest`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Upload failed');
      }

      const result = await response.json();
      console.log('Upload successful:', result);
      onUploadSuccess(); // Tell the parent component to switch views
      
    } catch (err) {
      console.error('Upload error:', err);
      setError(`Upload failed: ${err.message}`);
      setIsProcessing(false);
    }
  };

  return (
    <div className="file-upload-container">
      {isProcessing ? (
        <div>
          <p>Processing your document... This may take a moment.</p>
          {/* You could add a spinner animation here */}
        </div>
      ) : (
        <>
          <div
            className={`drop-zone ${isDragOver ? 'drag-over' : ''}`}
            onClick={handleDropZoneClick}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="application/pdf"
              style={{ display: 'none' }}
            />
            <p>Drag & drop a PDF here, or click to select a file.</p>
            {selectedFile && <p className="file-name">Selected: {selectedFile.name}</p>}
            {error && <p className="error-message">{error}</p>}
          </div>
          <button
            className="upload-button"
            onClick={handleUpload}
            disabled={!selectedFile}
          >
            Upload & Ingest
          </button>
        </>
      )}
    </div>
  );
};

export default FileUpload;