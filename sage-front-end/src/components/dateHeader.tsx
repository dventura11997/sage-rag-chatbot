import '../App.css'


const DateHeader: React.FC = () => {
    // Get the current date in AEST
    // Get the current date and time in AEST
    const currentDate = new Date().toLocaleDateString('en-AU', {
        timeZone: 'Australia/Sydney', // AEST timezone
        day: 'numeric',
        month: 'long',
        year: 'numeric',
    });

    const currentTime = new Date().toLocaleTimeString('en-AU', {
        timeZone: 'Australia/Sydney', // AEST timezone
        hour: 'numeric',
        minute: 'numeric',
        hour12: true, // Use 12-hour format with AM/PM
    });      
    return (
            <div className='date-header-container'>
                <text className='big-text black italic'>{currentDate}, {currentTime}</text>
                <div className='divider'></div>
            </div>
        )
    }

export default DateHeader;
