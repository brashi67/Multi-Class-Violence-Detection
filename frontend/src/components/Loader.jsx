import React from 'react'
import GridLoader from "react-spinners/GridLoader";

const Loader = () => {
  return (
      <div className="flex flex-col gap-5 justify-center items-center h-full">
          <GridLoader color="#36d7b7" size={46} />
          <div className='text-teal-100'>
              <div>Processing...</div>
              <div>This might take a while</div>
          </div>
      </div>
  );
}

export default Loader