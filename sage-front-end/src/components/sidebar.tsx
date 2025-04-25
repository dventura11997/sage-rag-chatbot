import '../App.css'
import Logo from '../assets/logo.png'
import Home from '../assets/home.png'
import Info from '../assets/info-circle.png'

export const Sidebar = () => { 
    return (
        <div className='sidebar-container'>
            <a href='/'><img src={Logo} alt="Logo" className='big'/></a>
            <a href='/' className='circle-tile'><img src={Home} alt="Home icon" className='icon'/></a>
            <a href='/info' className='circle-tile'><img src={Info} alt="Info icon" className='icon'/></a>
            
            
        </div>
    )    
}

export default Sidebar;