import '../App.css'
import LandingForm from '../components/landingForm';
import NexoLogo from '../assets/nexo-ai-logo.png'

function LandingPage() {
  return (
    <div className='landing-page'>
        <div className='lp-header-block'>
          <img src={NexoLogo} alt="Nexo AI Logo" className='logo'/>
          <div className='major-heading'>
            NEXO.AI
          </div>
        </div>
        <LandingForm />
    </div>
  ); 
}

export default LandingPage;