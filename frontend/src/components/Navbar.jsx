import React from 'react'

const Navbar = () => {
    return (
        <nav className="absolute top-0 left-0 w-screen">
            <div className="flex justify-center items-center h-16 bg-zinc-700 text-white relative shadow-sm font-mono">
                <h1 className="text-4xl font-extrabold">VIOCLASS</h1>
            </div>
        </nav>
    );
}

export default Navbar