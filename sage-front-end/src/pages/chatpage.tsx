import '../App.css'
import { useState } from 'react';
import ChatInput from '../components/chatInput';
import Sidebar from '../components/sidebar';
import DateHeader from '../components/dateHeader';

function ChatPage() {
  const [sharedInputValue] = useState("");

  return (
    <div>
        <section className='responsive-container'>
            <div className='left'>
              <Sidebar/>
            </div>
            <div className='middle'>
              <div className='chat-container'>
                <DateHeader/>
                <ChatInput initialInputValue={sharedInputValue} />
              </div>
            </div>
            <div className='right'></div>
        </section> 
    </div>
  ); 
}

export default ChatPage;