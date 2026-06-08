`timescale 1ns / 1ps

module unified_scratchpad_ldm #(
    parameter D_WIDTH      = 16,
    parameter MAC_LANES    = 8,
    parameter PACKED_W     = D_WIDTH * MAC_LANES, // 128-bit wide
    parameter ADDR_WIDTH   = 9,                   // 512 addresses (max 4096 values)
    parameter MEM_DEPTH    = 512
)(
    input  wire                 clk,
    input  wire                 we,            // Write Enable
    input  wire [ADDR_WIDTH-1:0] rd_addr,      // Dynamic Read Pointer from FSM
    input  wire [ADDR_WIDTH-1:0] wr_addr,      // Dynamic Write Pointer from FSM
    
    input  wire [D_WIDTH-1:0]    data_in,      // 16-bit result from the CE
    input  wire [2:0]            wr_lane_sel,  // Which of the 8 slots to write to
    
    output reg  [PACKED_W-1:0]   data_out_packed // 128-bit blast to the CE
);

    // The single, unified SRAM block mapped directly to BRAM
    reg [PACKED_W-1:0] unified_ram [0:MEM_DEPTH-1];
    reg [PACKED_W-1:0] temp_write_word;

    always @(posedge clk) begin
        // Port A: Read the 128-bit word for the 8 MACs
        data_out_packed <= unified_ram[rd_addr]; 
        
        // Port B: Read-Modify-Write the specific 16-bit output channel
        if (we) begin
            temp_write_word = unified_ram[wr_addr];
            temp_write_word[(wr_lane_sel * D_WIDTH) +: D_WIDTH] = data_in;
            unified_ram[wr_addr] <= temp_write_word;
        end
    end

endmodule