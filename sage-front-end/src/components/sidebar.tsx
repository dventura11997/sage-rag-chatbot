import '../App.css'
import NexoLogo from '../assets/nexo-ai-logo.png'
import Home from '../assets/house.png'
import Chat from '../assets/chat_icon.png'
import Back from '../assets/back.png'
import Settings from '../assets/setting.png'
import UserAvatar from '../assets/user-avatar.png'

export const Sidebar = () => { 
    return (
        <div className='sidebar'>
            <div className='sb-header-block'>
                <img src={NexoLogo} alt="Nexo AI Logo" className='logo small'/>
                <div className='sec-heading'>
                    NEXO.AI
                </div>
            </div>
            <div className='link-container'>
                <div className='link-container-grid'>
                    <img src={Home} alt="Home icon" className='icon-small'/>
                    <a href='/' className='navlink'>Home</a>
                    <img src={Chat} alt="Chat icon" className='icon-small'/>
                    <a href='/stock' className='navlink ch-hist'>Chat History</a>
                    <img src={Back} alt="Back icon" className='icon-small'/>
                    <a href='/' className='navlink'>Back</a>
                    <img src={Settings} alt="Settings icon" className='icon-small'/>
                    <a href='/' className='navlink'>Info</a>
                </div>
            </div>
            <div className='avatar-container'>
                <img src={UserAvatar} alt="UserAv icon" className='icon-med av-side'/>
                <div className='avatar-details'>
                    <div className='avatar-text-container'>
                        <h3 className='avatar-text'>John Smith</h3>
                    </div>
                    <h3 className='avatar-text title'>Manager</h3>
                </div>
            </div>
        </div>
    )    
}

export default Sidebar;