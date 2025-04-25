import { BrowserRouter, Routes, Route} from 'react-router-dom'
import LandingPage from './pages/landingpage'
import ErrorPage from './pages/errorpage'
import ChatPage from './pages/chatpage'
import InfoPage from './pages/infoPage'


function AppRoutes() {
  return (
    <div>
      <BrowserRouter>
        <Routes>
          <Route index element={<LandingPage />}/>
          <Route path='/' element={<LandingPage />}/>
          <Route path='/chat' element={<ChatPage />}/>
          <Route path='/info' element={<InfoPage />}/>
          <Route path='*' element={<ErrorPage />}/>
        </Routes>
      </BrowserRouter>
    </div>
  )
}

export default AppRoutes