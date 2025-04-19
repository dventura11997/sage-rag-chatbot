import { useEffect } from "react";
import { useNavigate } from "react-router-dom";

interface RedirectProps {
    page: any; // Define the type for the 'page' prop
    state?: any; // Define the type for the 'state' prop (optional)
}

const Redirect: React.FC<RedirectProps> = ({ page, state }) => {
    const navigate = useNavigate(); // This gives you the navigate function

    useEffect(() => {
        if (page) {
            navigate(`/${page}`, { state }); // Redirects to the given page
        }
    }, [page, state, navigate]);
    return null;
    
}

export default Redirect;