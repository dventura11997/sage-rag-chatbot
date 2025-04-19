import '../App.css'
import Redirect from './redirect';
import { useState } from 'react';

interface RedirectState {
    page: any; // Define the type for the 'page' prop
    state?: any; // Define the type for the 'state' prop (optional)
}


export const LandingForm = () => {
    const [redirectTo, setRedirectTo] = useState<RedirectState | null>(null);

    return (
        <div className='landing-page-container'>
            {redirectTo && <Redirect page={redirectTo.page} state={redirectTo.state} />}
            <form className='lp-form'>
                <input className='lp-form-field' placeholder='Enter email ID'></input>
                <button onClick={() => setRedirectTo({ page: 'chat' })} className='button-landing'>SSO Login</button>
            </form>
            
        </div>
    )     
}


export default LandingForm;