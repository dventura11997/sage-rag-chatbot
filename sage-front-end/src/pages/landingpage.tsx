import React, { useState } from 'react'; 
import '../App.css'
import Navbar from '../components/navbar';
import Redirect from '../utils/redirect';

interface RedirectState {
  page: any; // Define the type for the 'page' prop
}

function LandingPage() {
  const [redirectTo, setRedirectTo] = useState<RedirectState | null>(null);

  return (
    <div>
      {redirectTo && <Redirect page={redirectTo.page} />}
      <Navbar/>
      <section className='responsive-container'>
          <div className='left'>
          </div>
          <div className='middle'>
              <div className='container central'>
                <h1>Login To Get Started With</h1>
                <h1 className='primary-colour'>SAGE</h1>
                <h1>Below</h1>
                <button className='primary-button invert full margin' onClick={() => setRedirectTo({ page: 'chat' })}>Chat</button>
              </div>
          </div>
          <div className='right'></div>
      </section> 
</div>
  ); 
}

export default LandingPage;