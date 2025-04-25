import React from 'react';
import Info from '../assets/info-circle.png'
import { useNavigate } from 'react-router-dom';
import '../App.css';

const PrimaryButton: React.FC<{ label: string, page: string }> = ({ label, page }) => {
    const navigate = useNavigate();

    const handleClick = () => {
        navigate(`/${page}`);  // Redirect to the specified page
    };

    return (
        <button className="primary-button" onClick={handleClick}>
            <span className="medium-text">{label}</span>
            <img src={Info} alt="Info" className='logo'/>
        </button>
    )
    
}

export default PrimaryButton;