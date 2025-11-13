import taichi as ti
import numpy as np
from pyevtk.vtk import VtkFile, VtkRectilinearGrid,VtkUnstructuredGrid,VtkVertex
import os

@ti.kernel
def split_matrix_2D(arr:ti.template(),arr1:ti.template(),arr2:ti.template()):
    for p_i in ti.grouped(arr):
        arr1[p_i][0]=arr[p_i][0,0]
        arr1[p_i][1]=arr[p_i][0,1]
        arr2[p_i][0]=arr[p_i][1,0]
        arr2[p_i][1]=arr[p_i][1,1]

@ti.kernel
def set_v_scalar(v_scalar:ti.template(),v:ti.template()):
    for p_i in ti.grouped(v):
        v_scalar[p_i]=v[p_i].norm()

@ti.kernel
def split_matrix_3D(arr:ti.template(),arr1:ti.template(),arr2:ti.template(),arr3:ti.template()):
    for p_i in ti.grouped(arr):
        arr1[p_i][0]=arr[p_i][0,0]
        arr1[p_i][1]=arr[p_i][0,1]
        arr1[p_i][2]=arr[p_i][0,2]
        arr2[p_i][0]=arr[p_i][1,0]
        arr2[p_i][1]=arr[p_i][1,1]
        arr2[p_i][2]=arr[p_i][1,2]
        arr3[p_i][0]=arr[p_i][2,0]
        arr3[p_i][1]=arr[p_i][2,1]
        arr3[p_i][2]=arr[p_i][2,2]

def split_array(data,dim):
    if(dim==3):
        x,y,z=np.copy(data[:,0]),np.copy(data[:,1]),np.copy(data[:,2])
        return x,y,z
    elif(dim==2):
        x,y=np.copy(data[:,0]),np.copy(data[:,1])
        return x,y
    
def write_to_vtk(pos,scalar_data,vector_data,file_path,dim):
    # here points is regarded as unstructure-grid
    w = VtkFile(file_path, VtkUnstructuredGrid)
    npoints = pos.shape[0]
    
    w.openGrid()
    w.openPiece(ncells = npoints, npoints = npoints)
    
    # add points
    w.openElement("Points")
    if(dim==2):
        x,y=split_array(pos,dim)
        z=np.zeros_like(x,dtype=x.dtype)
    elif(dim==3):
        x,y,z=split_array(pos,dim)
    w.addData("points", (x,y,z))
    w.closeElement("Points")

    # add cell, now the cell is meaning less
    w.openElement("Cells")
    offsets = np.arange(start = 1, stop = npoints + 1, dtype = 'int32')   # index of last node in each cell
    connectivity = np.arange(npoints, dtype = 'int32')                    # each point is only connected to itself
    cell_types = np.empty(npoints, dtype = 'uint8') 
    cell_types[:] = VtkVertex.tid          
    w.addData("connectivity", connectivity)
    w.addData("offsets", offsets)
    w.addData("types", cell_types)
    w.closeElement("Cells")
    
    # add data
    scalar_keys = sorted(list(scalar_data.keys()))
    vector_keys = sorted(list(vector_data.keys()))
    if(len(scalar_keys)>0 and len(vector_keys)>0):             
        w.openData("Point", scalars = scalar_keys[0],vectors = vector_keys[0])
    elif(len(scalar_keys)>0):
        w.openData("Point", scalars = scalar_keys[0])
    elif(len(vector_keys)>0):
        w.openData("Point", vectors = vector_keys[0])
    else:
        w.openData("Point", scalars = "scalar")
    for key in scalar_keys:
        data = scalar_data[key]
        w.addData(key, data)
    for key in vector_keys:
        data = vector_data[key]
        if(dim==2):
            vx,vy=split_array(data,dim)
            vz=np.zeros_like(vx,dtype=vx.dtype)
        elif(dim==3):
            vx,vy,vz=split_array(data,dim)
        w.addData(key, (vx,vy,vz))
    w.closeData("Point")
    w.closePiece()
    w.closeGrid()

    w.appendData( (x,y,z) )
    w.appendData(connectivity).appendData(offsets).appendData(cell_types)
    for key in scalar_keys:
        data = scalar_data[key]
        w.appendData(data)
    for key in vector_keys:
        data = vector_data[key]
        if(dim==2):
            vx,vy=split_array(data,dim)
            vz=np.zeros_like(vx,dtype=vx.dtype)
        elif(dim==3):
            vx,vy,vz=split_array(data,dim)
        w.appendData((vx,vy,vz))
    w.save()
"""    
class Writer(Diff_Operator):
    def __init__(self,particle_system,root_folder):
        self.ps=particle_system
        self.index=0
        self.root_folder=root_folder
        if(not os.path.exists(root_folder)):
            os.makedirs(root_folder)
    
    def write(self,format="vtk"):
        # output x, v, div
        self.index+=1
        if(self.ps.debug==1):
            if(self.ps.dim==2):
                split_matrix_2D(self.ps.dPsi_use,self.ps.dPsi_x,self.ps.dPsi_y)
                split_matrix_2D(self.ps.dPsi_self,self.ps.dPsi_self_x,self.ps.dPsi_self_y)
                split_matrix_2D(self.ps.dPsi_tem,self.ps.dPsi_tem_x,self.ps.dPsi_tem_y)
            elif(self.ps.dim==3):
                split_matrix_3D(self.ps.dPsi_use,self.ps.dPsi_x,self.ps.dPsi_y,self.ps.dPsi_z)
                split_matrix_3D(self.ps.dPsi_self,self.ps.dPsi_self_x,self.ps.dPsi_self_y,self.ps.dPsi_self_z)
                split_matrix_3D(self.ps.dPsi_tem,self.ps.dPsi_tem_x,self.ps.dPsi_tem_y,self.ps.dPsi_tem_z)

        if(format=="vtk"):
            index_str=str(self.index).zfill(6)
            set_v_scalar(self.ps.v_scalar,self.ps.v)
            scalar_data={
                    "v_scalar":self.ps.v_scalar.to_numpy(),
                    "div":self.ps.div_error.to_numpy(),
                    "curl":self.ps.curl.to_numpy(),
                    "is_dynamic":self.ps.is_dynamic.to_numpy().astype(float),
                    "curvature":self.ps.curvature.to_numpy(),
                    "volume":self.ps.vor_regions_volume.to_numpy()
                }
            vector_data={
                    "velocity":self.ps.v.to_numpy(),
                    "curl_vector":self.ps.curl_vector.to_numpy(),
                    "norm":self.ps.norm.to_numpy(),
                    "diff":self.ps.diff_v.to_numpy()
                }
            if(self.ps.flow_map_calculation==2 or self.ps.flow_map_calculation==3):
                scalar_data["back_index"]=self.ps.traj_back_index.to_numpy()

            if(self.ps.debug==1):
                scalar_data["self.ps.CG_Jaccobi_pre"]=self.ps.CG_Jaccobi_pre.to_numpy()
                
                scalar_data["history"]=self.ps.has_been_surface.to_numpy().astype(float)
                scalar_data["pressure"]=self.ps.acc_p_use.to_numpy()
                
                scalar_data["free_surface_marker"]=self.ps.is_dynamic_surface_marker.to_numpy().astype(float)
                scalar_data["solid_marker"]=self.ps.is_dynamic_solid_marker.to_numpy().astype(float)
                scalar_data["r_div"]=self.ps.dPsi_div_self.to_numpy()
                scalar_data["r_div2"]=self.ps.dPsi_div_self2.to_numpy()
                scalar_data["r_div3"]=self.ps.dPsi_div_self3.to_numpy()
                scalar_data["r_div4"]=self.ps.dPsi_div_self4.to_numpy()
                scalar_data["color"]=self.ps.color_scalar.to_numpy()
                
                if(self.ps.solver_sub_type==2 or self.ps.solver_sub_type==3 or self.ps.solver_sub_type==4):
                    scalar_data["bad_cell"]=self.ps.vor_bad_cell.to_numpy().astype(float)

                vector_data["tem_grad_p"]=self.ps.tem_pressure_grad.to_numpy()
                vector_data["acc_use"]=self.ps.acc_v_use.to_numpy()
                vector_data["Psi_x"]=self.ps.dPsi_x.to_numpy()
                vector_data["Psi_y"]=self.ps.dPsi_y.to_numpy()
                vector_data["r_grad_x"]=self.ps.dPsi_self_x.to_numpy()
                vector_data["r_grad_y"]=self.ps.dPsi_self_y.to_numpy()
                vector_data["Psi_x_tem"]=self.ps.dPsi_tem_x.to_numpy()
                vector_data["Psi_y_tem"]=self.ps.dPsi_tem_y.to_numpy()
                vector_data["diff_x"]=self.ps.diff_x.to_numpy()

                
            write_to_vtk(
                self.ps.x.to_numpy(),
                scalar_data=scalar_data,
                vector_data=vector_data,
                file_path=os.path.join(self.root_folder,f"frame_{index_str}"),
                dim=self.ps.dim
            )
        else:
            raise NotImplementedError
        print(f"finish write frame {self.index}")

"""
if __name__ == "__main__":
    x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6.0]])
    v=np.array([[2.0,1.0,3],[2,0.9,4],[0.1,0.5,0.9],[4,5,6.0]])
    write_to_vtk(pos=x,scalar_data={},vector_data={"velocity":v},file_path=r".\test.vtu",dim=3)