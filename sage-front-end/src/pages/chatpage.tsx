import '../App.css'
import ChatInput from '../components/chatInput';
import Sidebar from '../components/sidebar';

function ChatPage() {
  return (
    <div>
        <section className='responsive-container'>
            <div className='left'>
              <Sidebar/>
            </div>
            <div className='middle'>
                <div>
                  <ChatInput />
                </div>
            </div>
            <div className='right'></div>
        </section> 
    </div>
  ); 
}

export default ChatPage;