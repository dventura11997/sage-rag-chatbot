import React from 'react';
import Send from '../assets/message-circle.png';

interface ChatTileProps {
    inputValue: string;
    setInputValue: (value: string) => void;
    handleSubmit: (e: React.FormEvent) => void;
}

const ChatTile: React.FC<ChatTileProps> = ({ inputValue, setInputValue, handleSubmit }) => {
    return (
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
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                handleSubmit(e);
                            }
                        }}
                        rows={1}
                    />
                </form>
            </div>
            <div>
                <text className="medium-text light border padding">How do I set up a new savings account?</text>
                <text className="medium-text light border padding">How can I increase my daily transfer limit?</text>
                <text className="medium-text light border padding">I lost my card — how do I block it and get a replacement?</text>
                <text className="medium-text light border padding">Where can I find my BSB and account number?</text>
                <text className="medium-text light border padding">How do I set up Apple Pay or Google Pay with my CommBank card?</text>
            </div>
        </div>
    );
};

export default ChatTile;