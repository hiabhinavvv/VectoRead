import React, { useState, useRef } from 'react';

const FileUpload = ({ onUploadSuccess }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const isValidFile = (file) => {
    if (!file) return false;

    const filename = file.name.toLowerCase();
    const validTypes = [
      'application/pdf',
      'text/csv',
      'application/vnd.ms-excel' // some browsers use this for csv
    ];

    const validExtensions = ['.pdf', '.csv'];

    return (
      validTypes.includes(file.type) ||
      validExtensions.some(ext => filename.endsWith(ext))
    );
  };

  const validateFile = (file) => {
    if (isValidFile(file)) {
      setSelectedFile(file);
      setError('');
      return true;
    }

    setSelectedFile(null);
    setError('Invalid file type. Please upload a PDF or CSV file.');
    return false;
  };

  const handleFileChange = (e) => validateFile(e.target.files[0]);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    validateFile(e.dataTransfer.files[0]);
  };

  const handleDropZoneClick = () => fileInputRef.current.click();

  const API_URL =
    import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

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

      if (result.session_id) {
        onUploadSuccess(result.session_id);
      } else {
        throw new Error('Session ID was not returned by the server');
      }
    } catch (err) {
      console.error('Upload error:', err);
      setError(`Upload failed: ${err.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="file-upload-container">
      {isProcessing ? (
        <div>
          <p>Processing your document… This may take a moment.</p>
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
              accept=".pdf,.csv"
              style={{ display: 'none' }}
            />

            <p>Drag & drop a PDF or CSV file here, or click to select.</p>

            {selectedFile && (
              <p className="file-name">
                Selected: {selectedFile.name}
              </p>
            )}

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
