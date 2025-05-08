import React, { useState, useEffect } from 'react';
import '../App.css';
import Send from '../assets/message-circle.png'
import chatSubmit from '../utils/chatSubmit';


//React.FC stands for "React Functional Component." It is a type provided by the React library that defines the structure of a functional component.
const ChatInput = ({ initialInputValue = "" }) => {
    // Declaring a state variable 'inputValue' to hold the current input value, initialized to an empty string
    const [inputValue, setInputValue] = useState(initialInputValue);
    const [messages, setMessages] = useState<{ text: string; type: 'user' | 'bot' }[]>([]); // Added type as user and bot. Array of strings

    
    // Add this useEffect at the top of your ChatInput component function
    useEffect(() => {
        const savedValue = localStorage.getItem('chatInputValue');
        if (savedValue) {
            setInputValue(savedValue);
            localStorage.removeItem('chatInputValue'); // Clear after using
            
            // Optional: Automatically submit the form with the retrieved value
            const syntheticEvent = { preventDefault: () => {} } as React.FormEvent;
            handleSubmit(syntheticEvent);
        }
    }, []);

    // Function to handle clicking on a suggestion
    const handleSuggestionClick = (suggestionText: string) => {
        // Set the input value to the suggestion text
        setInputValue(suggestionText);
        
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        // Checking if the input value is not just whitespace
        if (inputValue.trim()) {
            // Add the new user message to the messages array
            setMessages(prevMessages => [...prevMessages, { text: inputValue, type: 'user' }]);

            // Prepare the query body
            const requestBody = {
                query: inputValue, // User's input
                company: "Commonwealth Bank of Australia" // Static company name
            };

            // Call the formSubmit utility function
            const result = await chatSubmit('https://sage-rag-chatbot.onrender.com/chat_response_basic', requestBody);

            if (result.success) {
                // Add the chatbot's response to the messages array
                setMessages(prevMessages => [...prevMessages, 
                    { text: result.result?.message || "This is the chatbot's response.", type: 'bot' }, 
                    { text: result.result?.relevant_documents || "This is the chatbot's response.", type: 'bot' }
                ]);
            } else {
                // Handle error case
                setMessages(prevMessages => [...prevMessages, { text: result.error || "An error occurred.", type: 'bot' }]);
            }

            // Resetting the input value to an empty string after sending the message
            setInputValue("");
        }
    
    };

    // Common suggestions that users might want to ask
    const suggestions = [
        "How do I set up a new savings account?",
        "How can I increase my daily transfer limit?",
        "I lost my card — how do I block it and get a replacement?",
        "Where can I find my BSB and account number?",
        "How do I set up Apple Pay or Google Pay with my CommBank card?"
    ];

    return (
        <div className='chat-input-container'>
            <div className='messages-area'>
                {messages.map((msg, index) => (
                <div key={index} className='message-row'>
                    <div className={`chat-message ${msg.type}`}>
                    {msg.text}
                    </div>
                </div>
                ))}
            </div>
            <div className='chat-tile'>
                <div className='grid-2col'>
                    <img src={Send} alt="Send icon" className='icon'/>
                    <form onSubmit={handleSubmit}>
                    <textarea 
                        placeholder="Message Sage..." 
                        className="medium-text borderless full-width" 
                        value={inputValue} 
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) { // Submit on Enter, but not Shift + Enter
                                e.preventDefault(); // Prevent adding a new line
                                handleSubmit(e); // Trigger form submission
                            }
                        }}
                        rows={1}
                    />
                    </form>
                </div>
                <div>
                    {suggestions.map((suggestion, index) => (
                            <button 
                                key={index}
                                className="medium-text light border padding pointer no-bgd"
                                onClick={() => {
                                    handleSuggestionClick(suggestion);
                                }}
                            >
                                {suggestion}
                            </button>
                        ))}
                </div>
            </div>
        </div>        
    );
};

export default ChatInput;