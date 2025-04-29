import '../App.css'
import Sidebar from '../components/sidebar';
import ChatTile from '../components/chatTile';
import { useState } from 'react';

function InfoPage() {
    const [inputValue, setInputValue] = useState("");
    
    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        console.log("Submitted:", inputValue);
        setInputValue("");
    };

  return (
    <div>
        <section className='responsive-container'>
            <div className='left'>
              <Sidebar/>
            </div>
            <div className='middle'>
              <div className='info-container'>
                <div className='header-container'>
                    <h1>What Would You Like To Know About</h1>
                    <h1 className='primary-colour'>Commonwealth Bank</h1>
                </div>
                <ChatTile 
                    inputValue={inputValue}
                    setInputValue={setInputValue}
                />
                <div className='disclaimer-container'>
                    <text className='medium-text centered'>
                        SAGE is a GenAI Retrieval Augmented Generation (RAG) tool which aims to use a companies PDF documents to assist with customer requests. 
                        SAGE is currently configured with publicly available documents from the Commonwealth Bank of Australia (CBA)
                        SAGE might make some mistakes. By interacting with Sage.ai, you agree to the Ts & Cs.
                    </text>
                </div>
              </div>
              
            </div>
            <div className='right'></div>
        </section> 
    </div>
  ); 
}

export default InfoPage;