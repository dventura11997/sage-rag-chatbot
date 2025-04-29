import React, { useState, useEffect }  from 'react';
import Redirect from '../utils/redirect';

interface ChatTileProps {
    inputValue: string;
    setInputValue: (value: string) => void;
}

interface RedirectState {
    page: any; // Define the type for the 'page' prop
  }

const ChatTile: React.FC<ChatTileProps> = ({ inputValue, setInputValue }) => {
    const [redirectTo, setRedirectTo] = useState<RedirectState | null>(null);
  
    useEffect(() => {
        const savedValue = localStorage.getItem('chatInputValue');
        if (savedValue) {
            setInputValue(savedValue);
            localStorage.removeItem('chatInputValue'); // Clear after using
        }
    }, []);

    // Common suggestions that users might want to ask
    const suggestions = [
        "How do I set up a new savings account?",
        "How can I increase my daily transfer limit?",
        "I lost my card — how do I block it and get a replacement?",
        "Where can I find my BSB and account number?",
        "How do I set up Apple Pay or Google Pay with my CommBank card?"
    ];

    return (
        <div className='chat-tile'>
            {redirectTo && <Redirect page={redirectTo.page} />}
            <div>
                {suggestions.map((suggestion, index) => (
                        <button 
                            key={index}
                            className="medium-text light border padding pointer no-bgd"
                            onClick={() => {
                                setInputValue(suggestion); // This is correct - using the setter function
                                localStorage.setItem('chatInputValue', suggestion); // Store the value in localStorage
                                setRedirectTo({ page: 'chat' });
                            }}
                        >
                            {suggestion}
                        </button>
                    ))}
            </div>
        </div>
    );
};

export default ChatTile;