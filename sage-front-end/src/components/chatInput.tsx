import React, { useState } from 'react';
import '../App.css';
import Send from '../assets/send.png'
import UserAvatar from '../assets/user-avatar.png'
import AiAvatar from '../assets/ai-avatar.png'
import chatSubmit from '../utils/chatSubmit';


//React.FC stands for "React Functional Component." It is a type provided by the React library that defines the structure of a functional component.
const ChatInput = () => {
    // Declaring a state variable 'inputValue' to hold the current input value, initialized to an empty string
    const [inputValue, setInputValue] = useState("");
    const [messages, setMessages] = useState<{ text: string; type: 'user' | 'bot' }[]>([]); // Added type as user and bot. Array of strings

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        // Checking if the input value is not just whitespace
        if (inputValue.trim()) {
            // Add the new user message to the messages array
            setMessages(prevMessages => [...prevMessages, { text: inputValue, type: 'user' }]);
            // Add a placeholder response from the chatbot
            //setMessages(prevMessages => [...prevMessages, { text: "This is the chatbot's response.", type: 'bot' }]);

            // Prepare the query body
            const requestBody = {
                query: inputValue, // User's input
                company: "Australian Super" // Static company name
            };

            // Call the formSubmit utility function
            const result = await chatSubmit('http://127.0.0.1:5000/query_response', requestBody);

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

    return (
        <div className='chat-container'>
            <div className="chat-messages">
                    {messages.map((msg, index) => (
                            <div key={index} className={`chat-message ${msg.type}`}>
                                {msg.type === 'user' ? (
                                    <div className='messenger-role'>
                                        <img src={UserAvatar} alt="User Avatar" className='icon-med' />
                                        <h3 className='avatar-text'>{msg.type}</h3>
                                    </div>
                                ) : (
                                    <div className='messenger-role'>
                                        <img src={AiAvatar} alt="AI Avatar" className='icon-med' />
                                        <h3 className='avatar-text'>{msg.type}</h3>
                                    </div>
                                )}
                                {msg.text}
                            </div>
                    ))}
            </div>
            <div className='searchbar-container'>
                <form onSubmit={handleSubmit} className="chat-input-form">
                    <input
                        type="text"
                        placeholder="Type a message..."
                        className="chat-input"
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                    />
                    <button type="submit" className="button">
                        <p className='button-text'>Send</p>
                        <img src={Send} alt="Send icon" className='icon-send'/>
                    </button>
                </form>
            </div>
        </div>        
    );
};

export default ChatInput;