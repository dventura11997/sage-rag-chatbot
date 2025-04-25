import '../App.css'
import Logo from '../assets/logo.png'
import PrimaryButton from './primaryButton'

const Navbar = () =>
    {
        return (
            <nav className='navbar'>
                <a href='/'><img src={Logo} alt="Logo" className='logo big'/></a>
                <PrimaryButton label="Info" page="info" />
            </nav>

        )
    }

export default Navbar;
